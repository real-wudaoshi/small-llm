import math
from collections import Counter

from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def shannon_entropy(counter, total_count):
    if total_count <= 0:
        return 0.0
    entropy = 0.0
    for freq in counter.values():
        p = freq / total_count
        entropy -= p * math.log2(p)
    return entropy

tokenizer = Tokenizer.from_file("tokenizer.json")
dataset = load_dataset("roneneldan/TinyStories", split="validation")

total_samples = len(dataset)

token_counter = Counter()
total_tokens = 0

for i in tqdm(range(total_samples), total=total_samples):
    text = dataset[i]["text"] + "<eos>"
    token_ids = tokenizer.encode(text).ids
    token_counter.update(token_ids)
    total_tokens += len(token_ids)

unigram_entropy = shannon_entropy(token_counter, total_tokens)
unigram_perplexity = 2 ** unigram_entropy

print(f"unigram_entropy: {unigram_entropy:.6f}")
print(f"unigram_perplexity: {unigram_perplexity:.6f}")