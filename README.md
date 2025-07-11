# Ambiguity Resolution Fine-tuning Pipeline

A modular fine-tuning pipeline for training language models to resolve ambiguities in questions using the AmbigQA dataset. This project leverages Unsloth for efficient fine-tuning of large language models with LoRA (Low-Rank Adaptation).

## Overview

This pipeline processes ambiguous questions from the AmbigQA dataset and fine-tunes language models to generate clarifying questions that resolve ambiguities. The system is designed to identify unclear entities, timestamps, locations, and answer types in questions.

## Model Architecture

The pipeline supports multiple pre-quantized models through Unsloth:
- **Llama 3.1** (8B, 70B, 405B parameters)
- **Mistral** (7B, Nemo 12B)
- **Phi 3.5** (Mini, Medium)
- **Gemma 2** (9B, 27B)

All models are loaded with 4-bit quantization for memory efficiency and configured with LoRA for parameter-efficient fine-tuning.

## Unsloth Integration

[Unsloth](https://github.com/unslothai/unsloth) is used for:
- **4x faster downloading** of pre-quantized models
- **2x faster training** with optimized implementations
- **30% less VRAM usage** with gradient checkpointing
- **Native inference optimization** for deployment

## Requirements

### System Requirements
- CUDA-compatible GPU
- Python 3.8+
- CUDA toolkit

### Python Dependencies
```bash
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install torch torchvision torchaudio
pip install transformers datasets trl
pip install accelerate bitsandbytes
```

## Usage

### Basic Usage

```python
from model_manager import ModelManager
from data_processor import AmbigQAProcessor
from trainer_manager import TrainingManager
from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG

# Initialize components
model_manager = ModelManager(MODEL_CONFIG)
data_processor = AmbigQAProcessor(DATASET_CONFIG)
trainer_manager = TrainingManager(TRAINING_CONFIG)

# Load and configure model
model, tokenizer = model_manager.load_model()
model = model_manager.configure_lora()

# Process dataset
dataset = data_processor.process_full_pipeline(tokenizer)

# Train model
trainer = trainer_manager.create_trainer(model, tokenizer, dataset)
training_stats = trainer_manager.train()

# Prepare for inference
model_manager.prepare_for_inference()
```

### Inference Example

```python
from transformers import TextStreamer
from config import PromptTemplates

# Prepare input
prompt = PromptTemplates.ALPACA_TEMPLATE.format(
    PromptTemplates.AMBIGUITY_INSTRUCTION,
    "When did the Simpsons first air on television?",
    ""
)

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

# Generate response
text_streamer = TextStreamer(tokenizer)
outputs = model.generate(
    **inputs, 
    streamer=text_streamer, 
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7
)
```

## Memory Usage

Typical memory requirements:
- **8B Model**: ~6GB VRAM (4-bit quantization)
- **70B Model**: ~35GB VRAM (4-bit quantization)
- **Training**: +2-4GB additional VRAM
