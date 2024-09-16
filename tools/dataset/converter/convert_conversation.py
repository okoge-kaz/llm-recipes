import argparse
import json
import sys


def process_jsonl(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            data = json.loads(line)

            conversations = data.get("conversations", [])
            assert len(conversations) >= 2

            input_data = conversations[:-1]
            output_data = conversations[-1]

            data["input"] = input_data
            data["output"] = output_data

            json.dump(data, outfile)
            outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Process JSONL data")
    parser.add_argument("--input", help="Input JSONL file")
    parser.add_argument("--output", help="Output JSONL file")

    args = parser.parse_args()

    try:
        process_jsonl(args.input, args.output)
        print(f"Processing complete. Output written to {args.output}")
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
