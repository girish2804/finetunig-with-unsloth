"""
Data processing utilities for the AmbigQA dataset.
"""

import logging
from typing import Dict, List, Any, Optional
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from config import DatasetConfig, PromptTemplates

logger = logging.getLogger(__name__)


class AmbigQAProcessor:
    """Processes and formats the AmbigQA dataset for training."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset: Optional[Dataset] = None
        
    def load_dataset(self) -> Dataset:
        """
        Load the AmbigQA dataset from HuggingFace.
        
        Returns:
            Loaded dataset
            
        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            logger.info(f"Loading dataset: {self.config.DATASET_NAME}")
            
            dataset = load_dataset(
                self.config.DATASET_NAME,
                self.config.DATASET_CONFIG,
                split=self.config.SPLIT
            )
            
            logger.info(f"Dataset loaded successfully. Size: {len(dataset)}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise RuntimeError(f"Dataset loading failed: {str(e)}")
    
    def filter_ambiguous_questions(self, dataset: Dataset) -> Dataset:
        """
        Filter dataset to include only ambiguous questions.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Filtered dataset containing only ambiguous questions
        """
        try:
            logger.info("Filtering ambiguous questions")
            
            def is_ambiguous(example: Dict[str, Any]) -> bool:
                """Check if an example contains ambiguous questions."""
                annotations = example.get('annotations', {})
                annotation_type = annotations.get('type', [])
                
                return (
                    self.config.EXCLUDE_ANNOTATION_TYPE not in annotation_type and
                    self.config.INCLUDE_ANNOTATION_TYPE in annotation_type
                )
            
            filtered_dataset = dataset.filter(is_ambiguous)
            
            logger.info(f"Filtered dataset size: {len(filtered_dataset)}")
            return filtered_dataset
            
        except Exception as e:
            logger.error(f"Dataset filtering failed: {str(e)}")
            raise RuntimeError(f"Dataset filtering failed: {str(e)}")
    
    def extract_clarifying_questions(self, dataset: Dataset) -> Dataset:
        """
        Extract and combine clarifying questions from QA pairs.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with combined clarifying questions
        """
        try:
            logger.info("Extracting clarifying questions")
            
            def extract_questions(example: Dict[str, Any]) -> Dict[str, Any]:
                """Extract clarifying questions from an example."""
                qa_questions = []
                
                annotations = example.get('annotations', {})
                qa_pairs = annotations.get('qaPairs', [])
                
                if isinstance(qa_pairs, list):
                    for pair in qa_pairs:
                        if isinstance(pair, dict) and 'question' in pair:
                            questions = pair['question']
                            if isinstance(questions, list):
                                qa_questions.extend(questions)
                
                # Use the first clarifying question if available
                combined_questions = qa_questions[0] if qa_questions else ""
                example['combined_qa_questions'] = combined_questions
                
                return example
            
            processed_dataset = dataset.map(extract_questions)
            
            logger.info("Clarifying questions extracted successfully")
            return processed_dataset
            
        except Exception as e:
            logger.error(f"Question extraction failed: {str(e)}")
            raise RuntimeError(f"Question extraction failed: {str(e)}")
    
    def add_instructions(self, dataset: Dataset) -> Dataset:
        """
        Add task instructions to each example.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with instructions added
        """
        try:
            logger.info("Adding task instructions")
            
            def add_instruction(example: Dict[str, Any]) -> Dict[str, Any]:
                """Add instruction to an example."""
                example['instruction'] = PromptTemplates.AMBIGUITY_INSTRUCTION
                return example
            
            instructed_dataset = dataset.map(add_instruction)
            
            logger.info("Instructions added successfully")
            return instructed_dataset
            
        except Exception as e:
            logger.error(f"Instruction addition failed: {str(e)}")
            raise RuntimeError(f"Instruction addition failed: {str(e)}")
    
    def format_for_training(
        self, 
        dataset: Dataset, 
        tokenizer: PreTrainedTokenizer
    ) -> Dataset:
        """
        Format dataset for training using the Alpaca prompt template.
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer for adding special tokens
            
        Returns:
            Formatted dataset ready for training
        """
        try:
            logger.info("Formatting dataset for training")
            
            eos_token = tokenizer.eos_token
            if eos_token is None:
                logger.warning("EOS token not found in tokenizer, using default")
                eos_token = "</s>"
            
            def format_examples(examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
                """Format multiple examples for training."""
                instructions = examples["instruction"]
                inputs = examples["question"]
                outputs = examples['combined_qa_questions']
                
                formatted_texts = []
                
                for instruction, input_text, output_text in zip(instructions, inputs, outputs):
                    formatted_text = PromptTemplates.ALPACA_TEMPLATE.format(
                        instruction, input_text, output_text
                    ) + eos_token
                    
                    formatted_texts.append(formatted_text)
                
                return {"text": formatted_texts}
            
            formatted_dataset = dataset.map(
                format_examples,
                batched=True,
                num_proc=self.config.DATASET_NUM_PROC
            )
            
            logger.info("Dataset formatting completed successfully")
            return formatted_dataset
            
        except Exception as e:
            logger.error(f"Dataset formatting failed: {str(e)}")
            raise RuntimeError(f"Dataset formatting failed: {str(e)}")
    
    def process_full_pipeline(
        self, 
        tokenizer: PreTrainedTokenizer,
        limit_samples: Optional[int] = None
    ) -> Dataset:
        """
        Execute the complete data processing pipeline.
        
        Args:
            tokenizer: Tokenizer for formatting
            limit_samples: Optional limit on number of samples
            
        Returns:
            Fully processed dataset ready for training
        """
        try:
            logger.info("Starting full data processing pipeline")
            
            # Load dataset
            dataset = self.load_dataset()
            
            # Apply processing steps
            dataset = self.filter_ambiguous_questions(dataset)
            dataset = self.extract_clarifying_questions(dataset)
            dataset = self.add_instructions(dataset)
            dataset = self.format_for_training(dataset, tokenizer)
            
            # Limit samples if specified
            if limit_samples is not None:
                dataset = dataset.select(range(min(limit_samples, len(dataset))))
                logger.info(f"Limited dataset to {len(dataset)} samples")
            
            self.dataset = dataset
            
            logger.info("Data processing pipeline completed successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Data processing pipeline failed: {str(e)}")
            raise RuntimeError(f"Data processing pipeline failed: {str(e)}")
    
    def get_sample_text(self, index: int = 0) -> str:
        """
        Get a sample formatted text for inspection.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Formatted text sample
        """
        if self.dataset is None:
            raise RuntimeError("No dataset processed. Run process_full_pipeline() first.")
        
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        
        return self.dataset[index]['text']
    
    @property
    def dataset_info(self) -> Dict[str, Any]:
        """Get information about the processed dataset."""
        if self.dataset is None:
            return {"status": "No dataset processed"}
        
        return {
            "size": len(self.dataset),
            "columns": self.dataset.column_names,
            "features": str(self.dataset.features)
        }
