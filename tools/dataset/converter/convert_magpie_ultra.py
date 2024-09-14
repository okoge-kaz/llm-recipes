import argparse
import json
import sys


def process_input(input_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}", file=sys.stderr)
    return data


def convert_to_output(input_data, include_english: bool = False):
    output_data = []
    for item in input_data:
        if item.get("quality") in ["average", "good", "excellent"]:
            output_item = {
                "input": [
                    {
                        "role": "user",
                        "content": item["processed_translated_instruction"]
                    }
                ],
                "output": {
                    "role": "assistant",
                    "content": item["processed_translated_response"]
                },
                "quality": item["quality"],
                "primary_tag": item["primary_tag"],
            }
            output_data.append(output_item)
            if include_english:
                en_output_item = {
                    "input": [
                        {
                            "role": "user",
                            "content": item["instruction"],
                        }
                    ],
                    "output": {
                        "role": "assistant",
                        "content": item["response"]
                    },
                    "quality": item["quality"],
                    "primary_tag": item["primary_tag"],
                }
                output_data.append(en_output_item)

    return output_data


def save_output(output_data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in output_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Convert input JSONL to output JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--include-english", action="store_true")

    args = parser.parse_args()

    try:
        input_data = process_input(args.input)
        output_data = convert_to_output(
            input_data=input_data, include_english=args.include_english
        )
        save_output(output_data, args.output)
        print(f"Conversion completed. Output saved to {args.output}")
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
