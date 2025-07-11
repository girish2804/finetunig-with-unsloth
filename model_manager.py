"""
Model management utilities for loading and configuring language models.
"""

import torch
import logging
from typing import Tuple, Optional, Union
from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from config import (
    ModelConfig, 
    SupportedModels, 
    PrecisionType
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, configuration, and inference setup."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_name: Optional[str] = None
        
    def load_model(
        self, 
        model_name: str = SupportedModels.DEFAULT_MODEL,
        huggingface_token: Optional[str] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pre-trained model with specified configuration.
        
        Args:
            model_name: Name of the model to load
            huggingface_token: Optional HuggingFace token for gated models
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ValueError: If model loading fails
            RuntimeError: If CUDA is not available when required
        """
        try:
            self._validate_model_name(model_name)
            self._check_cuda_availability()
            
            logger.info(f"Loading model: {model_name}")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.config.MAX_SEQUENCE_LENGTH,
                dtype=self._get_optimal_dtype(),
                load_in_4bit=self.config.LOAD_IN_4BIT,
                token=huggingface_token
            )
            
            self.model_name = model_name
            logger.info("Model loaded successfully")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")
    
    def configure_lora(self) -> PreTrainedModel:
        """
        Configure LoRA (Low-Rank Adaptation) for the loaded model.
        
        Returns:
            Model with LoRA configuration applied
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        try:
            logger.info("Configuring LoRA adaptation")
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.LORA_RANK,
                target_modules=self.config.TARGET_MODULES,
                lora_alpha=self.config.LORA_ALPHA,
                lora_dropout=self.config.LORA_DROPOUT,
                bias=self.config.LORA_BIAS,
                use_gradient_checkpointing=self.config.USE_GRADIENT_CHECKPOINTING,
                random_state=self.config.RANDOM_STATE,
                use_rslora=self.config.USE_RSLORA,
                loftq_config=None,
            )
            
            logger.info("LoRA configuration applied successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"LoRA configuration failed: {str(e)}")
            raise RuntimeError(f"LoRA configuration failed: {str(e)}")
    
    def prepare_for_inference(self) -> None:
        """
        Prepare the model for inference mode.
        
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        try:
            logger.info("Preparing model for inference")
            FastLanguageModel.for_inference(self.model)
            logger.info("Model ready for inference")
            
        except Exception as e:
            logger.error(f"Inference preparation failed: {str(e)}")
            raise RuntimeError(f"Inference preparation failed: {str(e)}")
    
    def get_memory_stats(self) -> dict:
        """
        Get current GPU memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            gpu_stats = torch.cuda.get_device_properties(0)
            current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            
            return {
                "gpu_name": gpu_stats.name,
                "current_memory_gb": current_memory,
                "max_memory_gb": max_memory,
                "memory_utilization": round((current_memory / max_memory) * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}")
            return {"error": str(e)}
    
    def _validate_model_name(self, model_name: str) -> None:
        """Validate that the model name is supported."""
        if model_name not in SupportedModels.FOURBIT_MODELS:
            logger.warning(f"Model {model_name} is not in the list of pre-quantized models")
    
    def _check_cuda_availability(self) -> None:
        """Check if CUDA is available for GPU acceleration."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Training will be slower on CPU.")
    
    def _get_optimal_dtype(self) -> Optional[torch.dtype]:
        """
        Determine optimal data type based on hardware capabilities.
        
        Returns:
            Optimal torch dtype or None for auto-detection
        """
        if torch.cuda.is_available():
            # Check GPU compute capability for bfloat16 support
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:  # Ampere+ architecture
                return torch.bfloat16
            else:  # Older architectures
                return torch.float16
        
        return None  # Auto-detection for CPU
    
    @property
    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded."""
        return self.model is not None and self.tokenizer is not None
    
    @property
    def model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "No model loaded"}
        
        return {
            "model_name": self.model_name,
            "max_seq_length": self.config.MAX_SEQUENCE_LENGTH,
            "precision": "4bit" if self.config.LOAD_IN_4BIT else "full",
            "lora_rank": self.config.LORA_RANK,
            "target_modules": self.config.TARGET_MODULES
        }
