import sentencepiece as spm


def is_sentencepiece_model(file_path):
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(file_path)  # type: ignore
        return True
    except Exception:
        return False


file_path = '/bb/llm/gaf51275/hf-checkpoints/Phi-3-medium-4k-instruct/tokenizer.model'
if is_sentencepiece_model(file_path):
    print("The file is a SentencePiece model.")
else:
    print("The file is not a SentencePiece model.")
