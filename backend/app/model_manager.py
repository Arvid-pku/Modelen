"""
Model Manager: Handles loading and managing the LLM model.
"""

from typing import Optional, Dict, Any, Tuple
from functools import lru_cache
from collections import OrderedDict
import hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging
import time

from .hook_manager import HookManager, InterventionConfig as HookInterventionConfig
from .logit_lens import LogitLensDecoder

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache with max size and TTL support."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}

    def _make_key(self, *args, **kwargs) -> str:
        """Create a hash key from arguments."""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        if key not in self.cache:
            return None

        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            del self.cache[key]
            del self.timestamps[key]
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any):
        """Set item in cache, evicting LRU if needed."""
        if key in self.cache:
            del self.cache[key]

        self.cache[key] = value
        self.timestamps[key] = time.time()

        # Evict oldest if over max size
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.timestamps.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class ModelManager:
    """
    Manages the LLM model, tokenizer, and associated tools.

    Handles:
    - Model loading with appropriate precision
    - Device management (GPU/CPU)
    - Hook manager and logit lens setup
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.hook_manager = None
        self.logit_lens = None
        self.device = None
        self.model_name = None
        self._model_type = None

        # Caches
        self.tokenization_cache = LRUCache(max_size=200, ttl_seconds=600)
        self.inference_cache = LRUCache(max_size=50, ttl_seconds=300)

    def load_model(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        dtype: str = "float16",
        quantization: Optional[str] = None,
        trust_remote_code: bool = False
    ):
        """
        Load a model and set up associated components.

        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            dtype: Data type ('float16', 'bfloat16', 'float32')
            quantization: Quantization mode ('4bit', '8bit', or None)
            trust_remote_code: Whether to trust remote code for custom models
        """
        logger.info(f"Loading model: {model_name}")

        # Clean up previous model if exists
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Determine device
        if not device:  # Handle None or empty string
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        # Don't use float16 on CPU (unless quantized)
        if self.device == "cpu" and torch_dtype != torch.float32 and not quantization:
            torch_dtype = torch.float32
            logger.info("Using float32 for CPU inference")

        # Load config with attention output enabled
        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            output_attentions=True,
            output_hidden_states=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare quantization config
        quantization_config = None
        if quantization and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                if quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("Using 4-bit quantization")
                elif quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    logger.info("Using 8-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")
                quantization_config = None

        # Load model
        load_kwargs = {
            "config": self.config,
            "trust_remote_code": trust_remote_code,
        }

        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            if self.device == "cuda":
                load_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if self.device == "cpu" and not quantization_config:
            self.model = self.model.to(self.device)

        self.model.eval()
        self.model_name = model_name
        self.quantization = quantization

        # Detect model type
        self._model_type = self._detect_model_type()

        # Set up hook manager
        self.hook_manager = HookManager(self.model, model_type=self._model_type)

        # Set up logit lens
        self.logit_lens = LogitLensDecoder(
            self.model,
            self.tokenizer,
            model_type=self._model_type
        )

        logger.info(f"Model loaded on {self.device} with dtype {torch_dtype}, quantization={quantization}")

    def _detect_model_type(self) -> str:
        """Detect the model architecture type."""
        model_name_lower = self.model_name.lower()
        class_name_lower = self.model.__class__.__name__.lower()

        # Check both model name and class name
        combined = f"{model_name_lower} {class_name_lower}"

        if 'gpt2' in combined:
            return 'gpt2'
        elif 'gptneo' in combined or 'gpt-neo' in combined:
            return 'gptneo'
        elif 'gptj' in combined or 'gpt-j' in combined:
            return 'gptj'
        elif 'llama' in combined:
            return 'llama'
        elif 'mistral' in combined:
            return 'mistral'
        elif 'mixtral' in combined:
            return 'mixtral'
        elif 'qwen' in combined:
            return 'qwen'
        elif 'phi' in combined:
            return 'phi'
        elif 'gemma' in combined:
            return 'gemma'
        elif 'falcon' in combined:
            return 'falcon'
        elif 'opt' in combined:
            return 'opt'
        elif 'bloom' in combined:
            return 'bloom'
        else:
            return 'auto'

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "No model loaded"}

        return {
            "model_name": self.model_name,
            "num_layers": self.config.num_hidden_layers,
            "num_heads": self.config.num_attention_heads,
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "device": str(self.device)
        }

    def tokenize(self, text: str, use_cache: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize input text with optional caching."""
        cache_key = f"{self.model_name}:{text}"

        if use_cache:
            cached = self.tokenization_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for tokenization: {text[:30]}...")
                # Return a copy on the correct device
                return {k: v.clone().to(self.device) for k, v in cached.items()}

        result = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        if use_cache:
            # Store CPU copy in cache
            self.tokenization_cache.set(cache_key, {k: v.cpu().clone() for k, v in result.items()})

        return result

    def decode_tokens(self, token_ids: torch.Tensor) -> list:
        """Decode token IDs to strings."""
        # Handle different tensor shapes
        ids = token_ids.squeeze()
        if ids.dim() == 0:  # Single token (scalar)
            return [self.tokenizer.decode([ids.item()])]
        return [self.tokenizer.decode([tid]) for tid in ids.tolist()]

    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "tokenization_cache": self.tokenization_cache.stats(),
            "inference_cache": self.inference_cache.stats()
        }

    def clear_caches(self):
        """Clear all caches."""
        self.tokenization_cache.clear()
        self.inference_cache.clear()
        logger.info("All caches cleared")


# Global model manager instance
model_manager = ModelManager()
