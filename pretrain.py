from accelerate import Accelerator
import torch
import torch.nn as nn
from model import SmallLlm
from torch.utils.data import DataLoader
import config
import tokenizers
from transformers import get_cosine_schedule_with_warmup
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
import datasets
import os
import shutil
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--num_proc", type=int, default=4)
parser.add_argument("--warmup_ratio", type=float, default=0.05)
parser.add_argument("--map_batch_size", type=int, default=1024)
args = parser.parse_args()

tokenizer = tokenizers.Tokenizer.from_file("tokenizer.json")

if tokenizer.get_vocab_size() != config.MAX_VOCAB_SIZE:
    raise ValueError("Vocabulary size mismatch")

model = SmallLlm(config.MAX_VOCAB_SIZE, config.DIM, config.FFN_DIM, config.HEADS, config.LAYERS)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)
accelerator = Accelerator(mixed_precision="bf16")

def mapping_function(batch):
    sequences = []
    for sample in batch["text"]:
        sample = sample + "<eos>"
        tmp = tokenizer.encode(sample).ids
        for i in range(0, len(tmp), config.WINDOW_SIZE + 1):
            end = i + config.WINDOW_SIZE + 1
            sequences.append(tmp[i:end])
    return {
        "sequences": sequences
    }

def collate_fn(batch):
    start_time = time.perf_counter()
    sequences = [item["sequences"] for item in batch]
    padded = pad_sequence(sequences, batch_first=True, padding_value=config.PAD_ID)
    input_ids = padded[:, :-1]
    attention_mask = input_ids != config.PAD_ID
    labels = padded[:, 1:]
    end_time = time.perf_counter()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "time": end_time - start_time
    }

with accelerator.main_process_first():

    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")

    num_deleted = train_dataset.cleanup_cache_files() + val_dataset.cleanup_cache_files()
    print(f"Deleted {num_deleted} cache files")

    train_dataset = train_dataset.map(
        mapping_function,
        batched=True,
        batch_size=args.map_batch_size,
        remove_columns=train_dataset.column_names,
        num_proc=args.num_proc,
    )
    val_dataset = val_dataset.map(
        mapping_function,
        batched=True,
        batch_size=args.map_batch_size,
        remove_columns=val_dataset.column_names,
        num_proc=args.num_proc,
    )

    train_dataset.set_format("torch", columns=["sequences"], format_kwargs={"dtype": torch.long})
    val_dataset.set_format("torch", columns=["sequences"], format_kwargs={"dtype": torch.long})

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=args.num_proc
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=args.num_proc
)

total_steps = args.epoch * len(train_dataloader)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=math.ceil(args.warmup_ratio * total_steps),
    num_training_steps=total_steps
)
writer = SummaryWriter(log_dir="logs")
entropy_loss = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)

model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, scheduler
)

model.train()

global_step = 0

val_step = len(val_dataloader) * 100

def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

total_collate_time = 0
total_forward_time = 0
total_backward_time = 0
last_step = 0

for epoch in range(args.epoch):
    total_loss = 0
    for batch in tqdm(train_dataloader, total=len(train_dataloader)):
        global_step += 1
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        total_collate_time += batch["time"]
        t0 = get_time()

        outputs = model(input_ids, attention_mask)
        loss = entropy_loss(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
        total_loss += loss.item()

        t1 = get_time()
        total_forward_time += t1 - t0

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()    
        scheduler.step()

        t2 = get_time()
        total_backward_time += t2 - t1

        writer.add_scalar("loss/train", loss.item(), global_step)

        if accelerator.is_main_process and global_step % 1000 == 0:
            raw_model = accelerator.unwrap_model(model)
            torch.save(utils.clean_state_dict(raw_model.state_dict()), f"model.pth")
        if accelerator.is_main_process and global_step % 50 == 0:
            print(f"Collapsed time: {total_collate_time / (global_step - last_step)}")
            print(f"Forward time: {total_forward_time / (global_step - last_step)}")
            print(f"Backward time: {total_backward_time / (global_step - last_step)}")
            last_step = global_step
            total_collate_time = 0
            total_forward_time = 0
            total_backward_time = 0

        if val_dataloader is not None and val_step > 0 and global_step % val_step == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    outputs = model(input_ids, attention_mask)
                    loss = entropy_loss(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
                    val_loss += loss.item()
                writer.add_scalar("loss/val", val_loss / len(val_dataloader), global_step)
            model.train()
    accelerator.wait_for_everyone()
    print(f"Epoch {epoch + 1} finished, loss: {total_loss / len(train_dataloader)}")
    if accelerator.is_main_process:
        raw_model = accelerator.unwrap_model(model)
        torch.save(utils.clean_state_dict(raw_model.state_dict()), f"model.pth")

writer.close()