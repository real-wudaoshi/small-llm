from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
import config

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    special_tokens=config.SPECIAL_TOKENS,
    vocab_size=config.MAX_VOCAB_SIZE
)

dataset = load_dataset("roneneldan/TinyStories", split="train")

def iter_dataset(dataset):
    for sample in dataset:
        yield sample["text"]

tokenizer.train_from_iterator(iter_dataset(dataset), trainer)
tokenizer.save("tokenizer.json")