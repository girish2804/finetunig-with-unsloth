"""
Configuration constants for the Ambiguity Resolution Fine-tuning Pipeline.
"""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass


class ModelSize(Enum):
    """Enumeration for different model sizes."""
    MINI = "mini"
    SMALL = "7b"
    MEDIUM = "8b"
    LARGE = "70b"
    XLARGE = "405b"


class ModelFamily(Enum):
    """Enumeration for different model families."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    GEMMA = "gemma"


class PrecisionType(Enum):
    """Enumeration for model precision types."""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT4 = "4bit"
    INT8 = "8bit"


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    MAX_SEQUENCE_LENGTH: int = 1024
    LOAD_IN_4BIT: bool = True
    USE_GRADIENT_CHECKPOINTING: str = "unsloth"
    RANDOM_STATE: int = 3407
    
    # LoRA Configuration
    LORA_RANK: int = 16
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.0
    LORA_BIAS: str = "none"
    USE_RSLORA: bool = False
    
    # Target modules for LoRA adaptation
    TARGET_MODULES: List[str] = None
    
    def __post_init__(self):
        if self.TARGET_MODULES is None:
            self.TARGET_MODULES = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    BATCH_SIZE_PER_DEVICE: int = 2
    GRADIENT_ACCUMULATION_STEPS: int = 4
    WARMUP_STEPS: int = 5
    NUM_EPOCHS: int = 3
    MAX_TRAINING_STEPS: int = 500
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 0.01
    SCHEDULER_TYPE: str = "linear"
    OPTIMIZER: str = "adamw_8bit"
    LOGGING_STEPS: int = 1
    OUTPUT_DIR: str = "outputs"
    REPORT_TO: str = "none"
    DATASET_NUM_PROC: int = 2
    PACKING: bool = False


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    DATASET_NAME: str = "ambig_qa"
    DATASET_CONFIG: str = "light"
    SPLIT: str = "train"
    QUESTION_SEPARATOR: str = " , "
    MAX_GENERATION_TOKENS: int = 128
    
    # Filter criteria
    EXCLUDE_ANNOTATION_TYPE: str = "singleAnswer"
    INCLUDE_ANNOTATION_TYPE: str = "multipleQAs"


class SupportedModels:
    """Pre-quantized model identifiers."""
    
    FOURBIT_MODELS = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
        "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",
    ]
    
    DEFAULT_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


class PromptTemplates:
    """Template strings for different prompt formats."""
    
    ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    AMBIGUITY_INSTRUCTION = """In this task, you will receive a question that may contain ambiguities. First analyze the following aspects to find if there is any ambiguities according to the real-world facts:
- entities, objects, or events has multiple references or interpretations
- Unclear timestamps
- Unclear locations
- Unclear answer types (e.g., "When" refers to "which year or what date", and "Who" refers to "which person or which team")
If there is any ambiguities, you need to remove ambiguities by adding some clarifications to the question. Each clarification is an additional condition or explanations to the concept in the question that resolve its ambiguity.
- You are only allowed to add conditions or explanations, and you cannot change the original intent or semantics of the question.
- The conditions and explanations must be ground to real-word facts.
If there is no ambiguities, you only need to output the original question as it is."""


# Global configuration instances
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
DATASET_CONFIG = DatasetConfig()
