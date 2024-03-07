import argparse
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


def main() -> None:
    args = arg_parse()

    jsonl_data: list = []
    with open(args.input, "r") as f:
        for line in f:
            jsonl_data.append(json.loads(line))

    # shuffle
    import random

    print(f"shuffling len(jsonl_data)={len(jsonl_data)}")
    random.shuffle(jsonl_data)
    print(f"shuffled len(jsonl_data)={len(jsonl_data)}")

    # split
    split_ratio = 0.95
    split_index = int(len(jsonl_data) * split_ratio)
    train_data = jsonl_data[:split_index]
    val_data = jsonl_data[split_index:]

    print(f"len(train_data)={len(train_data)}, len(val_data)={len(val_data)}")

    # save
    with open(args.output + "/train.jsonl", "w") as f:
        for data in train_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    with open(args.output + "/val.jsonl", "w") as f:
        for data in val_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
