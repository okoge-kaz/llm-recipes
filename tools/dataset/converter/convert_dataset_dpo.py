import argparse
import json


def convert_jsonl(input_path: str, output_path: str, from_key: str, value_key: str) -> None:
    converted_data = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            conversations = item['conversations']
            chosen = item['chosen']
            rejected = item['rejected']
            converted_conversations = []
            for conversation in conversations:
                converted_conversation = {
                    'role': conversation[from_key],
                    'content': conversation[value_key]
                }
                converted_conversations.append(converted_conversation)
            converted_data.append({
                'conversations': converted_conversations,
                'chosen': chosen,
                'rejected': rejected
            })
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in converted_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL file keys to specified format.")
    parser.add_argument('--input-path', type=str, help='Path to the input JSONL file')
    parser.add_argument('--output-path', type=str, help='Path to the output JSONL file')
    parser.add_argument('--from-key', type=str, default='from', help='Key name to be converted to role')
    parser.add_argument('--value-key', type=str, default='value', help='Key name to be converted to context')

    args = parser.parse_args()

    convert_jsonl(args.input_path, args.output_path, args.from_key, args.value_key)

if __name__ == "__main__":
    main()
