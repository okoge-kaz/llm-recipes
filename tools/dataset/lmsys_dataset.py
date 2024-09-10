import argparse
import json
import hashlib
from typing import Any


def process_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
    conversation = sample.get("conversation", [])
    if len(conversation) < 2:
        return None

    user_message = conversation[0]
    assistant_message = conversation[1]

    if not user_message.get("content") or not assistant_message.get("content"):
        return None

    result = {
        "input": [{"role": "user", "content": user_message["content"]}],
        "output": {"role": "assistant", "content": assistant_message["content"]},
        "conversation": sample,
        "redacted": "NAME_" in user_message["content"] or "NAME_" in assistant_message["content"],
        "text": "user: " + user_message["content"] + "\n\nassistant: " + assistant_message["content"]
    }

    return result


def hash_sample(sample: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(sample, sort_keys=True).encode()).hexdigest()


def main(input_file: str, output_file: str, include_redacted: bool):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    processed_samples = []
    hash_set = set()
    invalid_count = 0
    redacted_count = 0
    non_redacted_count = 0

    for sample in data:
        processed = process_sample(sample)
        if processed:
            if processed["redacted"]:
                redacted_count += 1
                if include_redacted:
                    sample_hash = hash_sample(processed)
                    if sample_hash not in hash_set:
                        hash_set.add(sample_hash)
                        processed_samples.append(processed)
                    else:
                        print(f"Duplicate redacted sample found: {sample}")
            else:
                non_redacted_count += 1
                sample_hash = hash_sample(processed)
                if sample_hash not in hash_set:
                    hash_set.add(sample_hash)
                    processed_samples.append(processed)
                else:
                    print(f"Duplicate non-redacted sample found: {sample}")
        else:
            print(f"Invalid sample: {sample}")
            invalid_count += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in processed_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

    print(f"Processed {len(processed_samples)} unique samples.")
    print(f"Found {invalid_count} invalid samples.")
    print(f"Total samples: {len(data)}")
    print(f"Unique non-redacted samples: {non_redacted_count}")
    print(f"Redacted samples: {redacted_count}")
    if include_redacted:
        print("Redacted samples included in output")
    else:
        print("Redacted samples not included in output")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to specified JSONL format")
    parser.add_argument("--input-file", required=True, help="Input JSON file path")
    parser.add_argument("--output-file", required=True, help="Output JSONL file path")
    parser.add_argument("--include-redacted", action="store_true", help="Include redacted samples in output")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.include_redacted)
