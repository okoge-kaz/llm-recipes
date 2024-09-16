import argparse
import json
import copy


def convert_jsonl(input_path: str, output_path: str) -> None:
    converted_data = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            messages = item['messages']

            assert len(messages) % 2 == 0
            conversation_turn: int = len(messages) // 2

            inputs = []
            for i in range(conversation_turn):
                user_message = messages[i * 2]
                assistant_message = messages[i * 2 + 1]
                inputs.append(user_message)
                converted_data.append({
                    "input": copy.deepcopy(inputs),
                    "output": assistant_message
                })
                inputs.append(assistant_message)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in converted_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL file keys to specified format.")
    parser.add_argument('--input-path', type=str, help='Path to the input JSONL file')
    parser.add_argument('--output-path', type=str, help='Path to the output JSONL file')

    args = parser.parse_args()

    convert_jsonl(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
