from tokenizers import Tokenizer
from model import SmallLlm
import torch
import config

tokenizer = Tokenizer.from_file("tokenizer.json")
model = SmallLlm(config.MAX_VOCAB_SIZE, config.DIM, config.FFN_DIM, config.HEADS, config.LAYERS).to("cuda:1")
model.load_state_dict(torch.load("model.pth", map_location="cuda:1"))
model.eval()

while True:
    text = input("Enter a prompt: ")
    tokens = tokenizer.encode(text).ids
    tokens = torch.tensor(tokens, dtype=torch.long, device="cuda:1")
    out = model.generate(tokens, temperature=0.7, top_p=0.7, top_k=100, max_tokens=200)
    print(tokenizer.decode(out))