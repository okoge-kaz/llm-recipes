import json
import argparse
import os

# 引数パーサーを設定
parser = argparse.ArgumentParser("llm-recipes")
parser.add_argument("--data-file-path", type=str, required=True)
args = parser.parse_args()

# JSONファイルを開く
with open(args.data_file_path, 'r') as file:
    data = json.load(file)

# 新しいファイル名を生成（拡張子を ".jsonl" に変更）
base, _ = os.path.splitext(args.data_file_path)
new_file_path = base + ".jsonl"

# 新しいJSONLファイルを作成
with open(new_file_path, 'w') as file:
    for entry in data:
        # 各JSONオブジェクトを一行に変換して書き込む
        json.dump(entry, file, ensure_ascii=False)
        file.write('\n')  # 改行を追加
