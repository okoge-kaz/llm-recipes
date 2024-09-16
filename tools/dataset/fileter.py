import argparse
import json
from typing import List, Dict


def process_jsonl(file_path: str, threshold: float) -> List[Dict]:
    filtered_data = []
    with open(file_path, "r") as file:
        for line in file:
            entry = json.loads(line)
            if "overall" not in entry["scores"]:
                continue

            if entry["scores"]["overall"] >= threshold:
                conversations = entry["conversations"]
                # Get all messages except the last assistant message
                input_messages = conversations[:-1]
                assert len(conversations) % 2 == 0
                # Get only the last assistant message
                output_message = conversations[-1]
                assert output_message["role"] == "assistant"
                assert type(output_message) is dict
                filtered_data.append({"input": input_messages, "output": output_message})
    return filtered_data


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL file based on score threshold")
    parser.add_argument("--input_file", type=str, help="Path to input JSONL file")
    parser.add_argument("--output_file", type=str, help="Path to output JSONL file")
    parser.add_argument("--threshold", type=int, default=4, help="Score threshold for filtering (default: 0.0)")

    args = parser.parse_args()

    filtered_data = process_jsonl(args.input_file, args.threshold)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for entry in filtered_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"Processed data has been written to {args.output_file}")


if __name__ == "__main__":
    main()
