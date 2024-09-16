import argparse
import json
import random
from pathlib import Path


def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def extract_random_lines(input_path, output_path, num_lines):
    total_lines = count_lines(input_path)

    if num_lines >= total_lines:
        print(
            f"Warning: Requested {num_lines} lines, but file only contains {total_lines} lines. Extracting all lines."
        )
        num_lines = total_lines

    selected_indices = set(random.sample(range(total_lines), num_lines))

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            if i in selected_indices:
                try:
                    # Verify that the line is valid JSON
                    json.loads(line.strip())
                    outfile.write(line)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {i+1}. Skipping.")
                selected_indices.remove(i)
            if not selected_indices:
                break


def main():
    parser = argparse.ArgumentParser(description="Extract specified number of random lines from a JSONL file.")
    parser.add_argument("--input-path", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-path", required=True, help="Path to the output JSONL file")
    parser.add_argument("--num-lines", type=int, required=True, help="Number of lines to extract")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    if not input_path.is_file():
        print(f"Error: '{input_path}' is not a file.")
        return

    if args.seed is not None:
        random.seed(args.seed)

    extract_random_lines(input_path, output_path, args.num_lines)
    print(f"Extracted {args.num_lines} random lines from '{input_path}' to '{output_path}'.")


if __name__ == "__main__":
    main()
