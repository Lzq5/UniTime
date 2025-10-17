from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from . import register_loader
from .base import BaseModelLoader
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor

@register_loader("qwen2-vl")
class Qwen2VLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        if load_model:
            if self.model_finetune_path is None:
                model = Qwen2VLMRForConditionalGeneration.from_pretrained(
                    self.model_local_path,
                    **self.loading_kwargs,
                ) 
            else:
                model = Qwen2VLMRForConditionalGeneration.from_pretrained(
                    self.model_finetune_path,
                    **self.loading_kwargs,
                ) 
        processor = Qwen2VLMRProcessor.from_pretrained(self.model_local_path)
        tokenizer = processor.tokenizer
        model.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)

        return model, tokenizer, processor, config