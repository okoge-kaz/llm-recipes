import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Input file path")
args = parser.parse_args()

with open(args.input, "r") as file:
    lines = [json.loads(line) for line in file]
    print(f"Number of lines: {len(lines)}")
    for line in lines:
        conversations = line["conversations"]
        assert isinstance(conversations, list)
        assert len(conversations) % 2 == 0
        for i in range(0, len(conversations), 2):
            assert conversations[i]["role"] == "user"
            assert conversations[i+1]["role"] == "assistant"
            assert len(conversations[i]["content"]) > 0
            assert len(conversations[i+1]["content"]) > 0
    print("All checks passed")
