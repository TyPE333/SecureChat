# src/workers/model_loader.py

import importlib.util
import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from common.config import config

class GPUModelLoader:
    """
    Loads a text LLM (Qwen2.5 Instruct) on GPU with:
    INT 8 quantization,
    Flash Attention2.
    Device mapping for single GPU.
    """

    def __init__(self, worker_id: str):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU worker requires CUDA device.")
        
        self.worker_id = worker_id
        self.model_name = config.MODEL_NAME
        self.quant_mode = config.MODEL_QUANTIZATION_MODE.lower()
        self.use_flash_attn = self._resolve_flash_attn()
        self.device = "cuda:0"

    def _resolve_flash_attn(self) -> bool:
        """Enable flash attention only if requested and installed."""
        if not config.ENABLE_FLASH_ATTENTION:
            return False

        if importlib.util.find_spec("flash_attn") is None:
            logging.warning("[loader] flash_attn not installed; falling back to eager attention")
            return False
        return True
    
    def _load_tokenizer(self):
        logging.info(f"[loader] Loading tokenizer for {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def _load_model(self):
        attn_impl = "flash_attention_2" if self.use_flash_attn else "eager"

        logging.info(
            f"[loader] Loading model {self.model_name} |"
            f"quant={self.quant_mode} | attn={attn_impl}"
        )

        # --------------------------------
        # INT8 Quantization path
        # --------------------------------
        if self.quant_mode == "int8":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": 0},
                torch_dtype=torch.float16,
                load_in_8bit=True,
                attn_implementation=attn_impl,
            )
        
       
        # --------------------------------
        # FP16 path
        # -------------------------------- 

        elif self.quant_mode == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": 0},
                torch_dtype=torch.float16,
                attn_implementation=attn_impl,
            )
        
        else:
            raise ValueError(f"Unsupported quantization mode: {self.quant_mode}")
        
        logging.info("[loader] Model loaded successfully in GPU memory.")
        return model
    
    def load(self):
        """Load tokenizer and model, cache on the instance, and return them."""
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        return self.tokenizer, self.model
