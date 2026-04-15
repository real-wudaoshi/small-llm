import argparse
import math
from collections import Counter

from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def shannon_entropy_from_counter(counter: Counter[int], total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    entropy = 0.0
    for freq in counter.values():
        p = freq / total_count
        entropy -= p * math.log2(p)
    return entropy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TinyStories token entropy.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer.json",
        help="Path to tokenizer json used for tokenization.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Max number of samples to process (0 means all).",
    )
    parser.add_argument(
        "--add_eos",
        action="store_true",
        help="Append '<eos>' to each sample before tokenization.",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    dataset = load_dataset("roneneldan/TinyStories", split=args.split)

    total_samples = len(dataset)
    if args.max_samples > 0:
        total_samples = min(total_samples, args.max_samples)

    token_counter: Counter[int] = Counter()
    total_tokens = 0

    iterator = range(total_samples)
    for i in tqdm(iterator, total=total_samples, desc="Tokenizing"):
        text = dataset[i]["text"]
        if args.add_eos:
            text += "<eos>"
        token_ids = tokenizer.encode(text).ids
        token_counter.update(token_ids)
        total_tokens += len(token_ids)

    entropy_bits_per_token = shannon_entropy_from_counter(token_counter, total_tokens)
    unigram_perplexity = 2 ** entropy_bits_per_token

    print(f"split: {args.split}")
    print(f"samples: {total_samples}")
    print(f"total_tokens: {total_tokens}")
    print(f"unique_tokens: {len(token_counter)}")
    print(f"entropy_bits_per_token: {entropy_bits_per_token:.6f}")
    print(f"unigram_perplexity: {unigram_perplexity:.6f}")


if __name__ == "__main__":
    main()
