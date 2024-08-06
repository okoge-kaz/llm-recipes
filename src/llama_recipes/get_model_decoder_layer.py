from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer


def get_model_decoder_layer(
    model_name: str,
) -> type[LlamaDecoderLayer] | type[MistralDecoderLayer] | type[Phi3DecoderLayer] | type[Gemma2DecoderLayer]:
    if "Llama" in model_name or "Swallow" in model_name or "Yi" in model_name:
        return LlamaDecoderLayer
    elif "Mistral" in model_name or "mistral" in model_name or "Codestral" in model_name:
        return MistralDecoderLayer
    elif "Phi-3" in model_name:
        return Phi3DecoderLayer
    elif "gemma-2" in model_name:
        return Gemma2DecoderLayer
    else:
        raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
