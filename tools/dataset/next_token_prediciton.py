import argparse
import json

def check_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            try:
                json_obj = json.loads(line)
                if json_obj.get('role') == 'next_token_prediction':
                    pass
                else:
                    print(f"Line {line_number}: 'role': 'next_token_prediction' not found")
            except json.JSONDecodeError:
                print(f"Line {line_number}: Invalid JSON")

def main():
    parser = argparse.ArgumentParser(description="Check JSONL file for 'role': 'next_token_prediction'")
    parser.add_argument('--file_path', help='Path to the JSONL file')
    args = parser.parse_args()

    check_jsonl_file(args.file_path)

if __name__ == '__main__':
    main()
