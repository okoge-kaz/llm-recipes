import argparse
import json
import sys
import random
import re


def is_empty_or_template(content):
    content = content.strip()
    return content in ("", "\n", "\n\n") or content in ("回答例:", "回答例；", "解答例:", "解答例；")


def clean_content_start(content):
    # Remove leading ">\n\n" or ">\n\n\n"
    content = re.sub(r"^>\n\n+", "", content)
    # Remove leading asterisks
    content = re.sub(r"^\s*\*+\s*", "", content)
    return content


def clean_content_end(content):
    # Remove leading newlines and spaces
    content = content.lstrip("\n ")

    # Process the end of the content
    lines = content.splitlines()
    if lines:
        # Clean the last line
        last_line = lines[-1].rstrip()
        # Remove trailing "**" if present
        last_line = re.sub(r"\*+\s*$", "", last_line)
        lines[-1] = last_line

    # Join the lines back together
    content = "\n".join(lines)

    # Remove trailing asterisks followed by newline
    content = re.sub(r"\*+\s*\n$", "\n", content)

    # Ensure the content ends with exactly one newline
    content = content.rstrip() + "\n"

    return content


def process_jsonl(input_file, output_file):
    processed_data = []
    seen_contents = set()
    with open(input_file, "r") as infile:
        for line in infile:
            try:
                data = json.loads(line)

                # Transform input
                if "input" in data:
                    data["input"] = [data["input"]]

                # Clean input and output content
                input_content = clean_content_end(clean_content_start(data["input"][0].get("content", "")))
                output_content = clean_content_end(clean_content_start(data.get("output", {}).get("content", "")))

                # Check for empty or template content
                if is_empty_or_template(input_content) or is_empty_or_template(output_content):
                    continue

                # Check for duplicates
                content_pair = (input_content, output_content)
                if content_pair in seen_contents:
                    continue
                seen_contents.add(content_pair)

                # Update cleaned contents
                data["input"][0]["content"] = input_content
                data["output"]["content"] = output_content

                # add text section
                data["text"] = "user: " + input_content + "\n" + "assistant: " + output_content

                processed_data.append(data)

            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}", file=sys.stderr)

    # Shuffle the processed data
    random.shuffle(processed_data)

    # Write the shuffled data to the output file
    with open(output_file, "w") as outfile:
        for data in processed_data:
            json.dump(data, outfile)
            outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Process and shuffle JSONL files")
    parser.add_argument("--input", help="Input JSONL file")
    parser.add_argument("--output", help="Output JSONL file")
    parser.add_argument("--seed", type=int, help="Random seed for shuffling", default=123)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    process_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
