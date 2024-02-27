import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser("chat template checker")
parser.add_argument("--tokenizer-dir")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=args.tokenizer_dir
)

conversations: dict = {
    "input": [
        {
            "role": "user",
            "content": "こんにちは！"
        },
        {
            "role": "assistant",
            "content": "こんにちは！ご質問やお困りのことがありましたら、何でもご相談ください。何が必要か教えてください。"
        },
        {
            "role": "user",
            "content": "世界のすべての国をアルファベット順に、それぞれの国の人口を教えてください。"
        }
    ],
    "output": "世界中の国をアルファベット順に並べたリストと、その国の推定人口です：\n\nアフガニスタン: 38,928,346 アルバニア: 2,877,797 アルジェリア：44,344,744 アンドラ: 77,265 アンゴラ: 32,878,272 アンティグア・バーブーダ: 97,929 アルゼンチン: 45,195,774 アルメニア: 2,977,600 オーストラリア: 25,499,884 オーストリア: 9,006,398 アゼルバイジャン: 10,134,604 バハマ：393,248 バーレーン: 1,714,571 バングラデシュ: 164,689,383164,689,383 バルバドス: 287,375 ベラルーシ: 9,449,323 ベルギー: 11,589,623 ベリーズ: 397,628 ベナン: 12,123,200 ブータン: 754,288 ボリビア: 11,673,095 ボスニア・ヘルツェゴビナ: 3,279,000 ボツワナ：2,335,814 ブラジル: 212,559,417 ブルネイ: 441,484 ブルガリア: 7,034,879 ブルキナファソ: 20,903,273 ブルンジ: 11,890,781 カボベルデ: 555,987 カンボジア: 16,205,218 カメルーン: 26,545,863 カナダ：37,742,154 中央アフリカ共和国4,829,767 チャド: 16,425,864 チリ: 19,116,201 中国: 1,439,323,776 コロンビア: 50,882,891 コモロ: 869,601 コンゴ民主共和国：87,534,403 コンゴ共和国5,457,821 コスタリカ5,094,118 コートジボワール: 26,378,274 クロアチア: 4,105,267 キューバ: 11,239,224 キプロス：1,207,359 チェコ：10,708,919 デンマーク：5,792,2025,792,202"
}

chat_template: str = "{{ bos_token }}<<SYS>>\\nあなたは誠実で優秀な日本人のアシスタントです。\\n<</SYS>>\\n\\n{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ '[ASST] ' + message['content'] + ' [/ASST]' + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

print("before apply chat template")

prompt: str = tokenizer.apply_chat_template(
    conversations["input"],  # type: ignore
    chat_template=chat_template,
    tokenize=False
)

print(prompt)
