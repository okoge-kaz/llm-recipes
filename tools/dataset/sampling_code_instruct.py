import argparse
import json
import random
from pathlib import Path


def sample_jsonl(input_file, output_file, sample_size):
    """
    JSONLファイルからランダムにサンプルを抽出し、新しいJSONLファイルに保存する。

    Args:
        input_file (str): 入力JSONLファイルのパス
        output_file (str): 出力JSONLファイルのパス
        sample_size (int): 抽出するサンプル数
    """
    # 入力ファイルが存在するか確認
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    # 全レコードを読み込む
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 空行を無視
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"警告: 不正なJSONデータを無視しました: {line}")

    # サンプルサイズが入力データのサイズを超えていないことを確認
    total_records = len(records)
    if sample_size > total_records:
        print(f"警告: 要求されたサンプル数({sample_size})が入力データの総数({total_records})を超えています。")
        print("すべてのレコードを返します。")
        sample_size = total_records

    # レコードをシャッフルし、サンプルを抽出
    random.shuffle(records)
    samples = records[:sample_size]

    # 出力ディレクトリが存在しない場合は作成
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # サンプルを出力ファイルに書き込む
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in samples:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"{total_records}レコード中{sample_size}レコードをサンプリングし、{output_file}に保存しました。")

def main():
    # コマンドライン引数の定義
    parser = argparse.ArgumentParser(description='JSONLファイルからランダムにサンプルを抽出します。')
    parser.add_argument('--input', '-i', required=True, help='入力JSONLファイルのパス')
    parser.add_argument('--output', '-o', required=True, help='出力JSONLファイルのパス')
    parser.add_argument('--sample_size', '-n', type=int, required=True, help='抽出するサンプル数')
    parser.add_argument('--seed', '-s', type=int, default=None, help='乱数シード（再現性のため）')

    args = parser.parse_args()

    # 乱数シードが指定されている場合は設定
    if args.seed is not None:
        random.seed(args.seed)

    # サンプリング実行
    sample_jsonl(args.input, args.output, args.sample_size)

if __name__ == "__main__":
    main()
