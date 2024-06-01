import sentencepiece as spm


def is_sentencepiece_model(file_path):
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(file_path)  # type: ignore
        return True
    except Exception:
        return False


file_path = '/home/ext_kazuki_fujii_rio_gsic_titech/hf_checkpoints/Codestral-22B-v0.1/tokenizer.model'
if is_sentencepiece_model(file_path):
    print("The file is a SentencePiece model.")
else:
    print("The file is not a SentencePiece model.")
