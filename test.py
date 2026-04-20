from tokenizers import Tokenizer
from model import SmallLlm
import torch
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file("tokenizer.json")
model = SmallLlm(config.MAX_VOCAB_SIZE, config.DIM, config.FFN_DIM, config.HEADS, config.LAYERS).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

while True:
    text = input("Enter a prompt: ")
    tokens = tokenizer.encode(text).ids
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    out = model.generate(tokens, temperature=0.5, top_p=0.7, top_k=30, max_tokens=200, repetition_penalty=1.20)
    print(tokenizer.decode(out))