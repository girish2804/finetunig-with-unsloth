"""
Training management utilities for fine-tuning language models.
"""

import logging
from typing import Dict, Any, Optional
from datasets import Dataset
from trl import SFTTrainer
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer, 
    TrainingArguments,
    TrainerCallback
)
from unsloth import is_bfloat16_supported

from config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages the training process for language model fine-tuning."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer: Optional[SFTTrainer] = None
        self.training_stats: Optional[Dict[str, Any]] = None
        
    def create_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        text_field: str = "text"
    ) -> SFTTrainer:
        """
        Create an SFTTrainer instance with the specified configuration.
        
        Args:
            model: Pre-trained model to fine-tune
            tokenizer: Tokenizer for the model
            train_dataset: Dataset for training
            text_field: Name of the text field in the dataset
            
        Returns:
            Configured SFTTrainer instance
            
        Raises:
            RuntimeError: If trainer creation fails
        """
        try:
            logger.info("Creating SFTTrainer")
            
            training_args = self._create_training_arguments()
            
            self.trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                dataset_text_field=text_field,
                max_seq_length=1024,  # This should match model config
                dataset_num_proc=self.config.DATASET_NUM_PROC,
                packing=self.config.PACKING,
                args=training_args,
            )
            
            logger.info("SFTTrainer created successfully")
            return self.trainer
            
        except Exception as e:
            logger.error(f"Failed to create trainer: {str(e)}")
            raise RuntimeError(f"Trainer creation failed: {str(e)}")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the training process.
        
        Returns:
            Training statistics
            
        Raises:
            RuntimeError: If training fails or no trainer is available
        """
        if self.trainer is None:
            raise RuntimeError("No trainer available. Call create_trainer() first.")
        
        try:
            logger.info("Starting training process")
            
            # Log training configuration
            self._log_training_config()
            
            # Execute training
            self.training_stats = self.trainer.train()
            
            logger.info("Training completed successfully")
            
            # Log training results
            self._log_training_results()
            
            return self.training_stats
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def save_model(self, output_path: str) -> None:
        """
        Save the trained model to the specified path.
        
        Args:
            output_path: Path where the model should be saved
            
        Raises:
            RuntimeError: If saving fails or no trainer is available
        """
        if self.trainer is None:
            raise RuntimeError("No trainer available. Call create_trainer() first.")
        
        try:
            logger.info(f"Saving model to: {output_path}")
            
            self.trainer.save_model(output_path)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise RuntimeError(f"Model saving failed: {str(e)}")
    
    def _create_training_arguments(self) -> TrainingArguments:
        """
        Create TrainingArguments with the specified configuration.
        
        Returns:
            Configured TrainingArguments instance
        """
        return TrainingArguments(
            per_device_train_batch_size=self.config.BATCH_SIZE_PER_DEVICE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=self.config.WARMUP_STEPS,
            num_train_epochs=self.config.NUM_EPOCHS,
            max_steps=self.config.MAX_TRAINING_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.LOGGING_STEPS,
            optim=self.config.OPTIMIZER,
            weight_decay=self.config.WEIGHT_DECAY,
            lr_scheduler_type=self.config.SCHEDULER_TYPE,
            seed=3407,  # Using same seed as model config for consistency
            output_dir=self.config.OUTPUT_DIR,
            report_to=self.config.REPORT_TO,
            save_strategy="epoch",
            evaluation_strategy="no",
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=None,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
    
    def _log_training_config(self) -> None:
        """Log the training configuration."""
        config_info = {
            "batch_size": self.config.BATCH_SIZE_PER_DEVICE,
            "gradient_accumulation_steps": self.config.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": self.config.LEARNING_RATE,
            "num_epochs": self.config.NUM_EPOCHS,
            "max_steps": self.config.MAX_TRAINING_STEPS,
            "optimizer": self.config.OPTIMIZER,
            "scheduler": self.config.SCHEDULER_TYPE,
            "weight_decay": self.config.WEIGHT_DECAY,
        }
        
        logger.info(f"Training configuration: {config_info}")
    
    def _log_training_results(self) -> None:
        """Log the training results."""
        if self.training_stats is None:
            return
        
        try:
            # Extract key metrics from training stats
            final_loss = self.training_stats.training_loss
            steps_completed = self.training_stats.global_step
            
            logger.info(f"Training completed - Final loss: {final_loss:.4f}, Steps: {steps_completed}")
            
        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not extract training metrics: {str(e)}")
    
    @property
    def is_trained(self) -> bool:
        """Check if training has been completed."""
        return self.training_stats is not None
    
    @property
    def training_info(self) -> Dict[str, Any]:
        """Get information about the training process."""
        if not self.is_trained:
            return {"status": "Not trained"}
        
        try:
            return {
                "training_loss": self.training_stats.training_loss,
                "global_step": self.training_stats.global_step,
                "epoch": self.training_stats.epoch,
                "training_completed": True
            }
        except (AttributeError, KeyError):
            return {"status": "Training completed but stats unavailable"}


class TrainingCallback(TrainerCallback):
    """Custom callback for monitoring training progress."""
    
    def __init__(self):
        self.step_count = 0
        self.losses = []