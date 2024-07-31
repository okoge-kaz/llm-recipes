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
    ]
}

conversations_with_output = [
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
    },
    {
        "role": "assistant",
        "content": "output",
    }
]

chat_template: str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

print("before apply chat template")

prompt: str = tokenizer.apply_chat_template(
    [{"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"}] + conversations["input"],  # type: ignore
    add_generation_prompt=True,
    # tokenize=False
)

print(prompt)
print(type(prompt))

print("--------------------------------")

print(tokenizer.apply_chat_template(
    conversation=conversations_with_output,
    # tokenize=False
))
