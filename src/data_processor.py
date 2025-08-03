"""
Data processing utilities for BaZi-GPT fine-tuning.
"""

import json
import pandas as pd
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .utils import clean_text, truncate_text, estimate_tokens, setup_logging


class BaZiDataProcessor:
    """Process BaZi Q&A data for fine-tuning."""
    
    def __init__(self, system_prompt_file: str = "config/system_prompt.txt"):
        """
        Initialize the data processor.
        
        Args:
            system_prompt_file: Path to system prompt file
        """
        self.logger = setup_logging()
        self.system_prompt = self._load_system_prompt(system_prompt_file)
        self.processed_data = []
        
    def _load_system_prompt(self, file_path: str) -> str:
        """Load system prompt from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            self.logger.info(f"Loaded system prompt from {file_path}")
            return prompt
        except FileNotFoundError:
            self.logger.warning(f"System prompt file not found: {file_path}")
            return self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt if file not found."""
        return """你是一位精通八字命理的專業顧問。請用中文回答問題，並提供英文翻譯。
        
        回答格式：
        1. 中文回答（詳細專業）
        2. English Translation
        3. 相關建議或注意事項
        
        請確保回答準確、專業，並包含適當的免責聲明。"""
    
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        Load training data from CSV file.
        
        Args:
            file_path: Path to CSV file with 'question' and 'answer' columns
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Validate required columns
            required_cols = ['question', 'answer']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning...")
        
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna(subset=['question', 'answer'])
        self.logger.info(f"Removed {initial_count - len(df)} rows with missing values")
        
        # Clean text fields
        df['question'] = df['question'].apply(clean_text)
        df['answer'] = df['answer'].apply(clean_text)
        
        # Remove empty responses
        df = df[df['question'].str.len() > 0]
        df = df[df['answer'].str.len() > 0]
        
        # Truncate long responses to prevent token limit issues
        df['answer'] = df['answer'].apply(lambda x: truncate_text(x, 3000))
        
        self.logger.info(f"Data cleaning complete. Final count: {len(df)} rows")
        return df
    
    def create_training_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert data to OpenAI fine-tuning format.
        
        Args:
            df: Input DataFrame with question/answer pairs
            
        Returns:
            List of training examples in OpenAI format
        """
        training_data = []
        
        for _, row in df.iterrows():
            example = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": row['question']},
                    {"role": "assistant", "content": row['answer']}
                ]
            }
            training_data.append(example)
        
        self.processed_data = training_data
        self.logger.info(f"Created {len(training_data)} training examples")
        return training_data
    
    def validate_training_data(self, data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate training data format and content.
        
        Args:
            data: Training data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        for i, example in enumerate(data):
            # Check required structure
            if "messages" not in example:
                issues.append(f"Example {i}: Missing 'messages' key")
                continue
            
            messages = example["messages"]
            if len(messages) != 3:
                issues.append(f"Example {i}: Should have exactly 3 messages")
                continue
            
            # Check message roles
            expected_roles = ["system", "user", "assistant"]
            for j, msg in enumerate(messages):
                if msg.get("role") != expected_roles[j]:
                    issues.append(f"Example {i}, Message {j}: Expected role '{expected_roles[j]}', got '{msg.get('role')}'")
                
                if "content" not in msg or not msg["content"].strip():
                    issues.append(f"Example {i}, Message {j}: Empty or missing content")
            
            # Check token count estimation
            total_tokens = sum(estimate_tokens(msg["content"]) for msg in messages)
            if total_tokens > 4000:
                issues.append(f"Example {i}: Estimated {total_tokens} tokens, may exceed limits")
        
        is_valid = len(issues) == 0
        self.logger.info(f"Validation complete. Valid: {is_valid}, Issues found: {len(issues)}")
        
        return is_valid, issues
    
    def save_training_file(self, data: List[Dict[str, Any]], output_path: str) -> str:
        """
        Save training data to JSONL file.
        
        Args:
            data: Training data to save
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(output_file, 'w') as writer:
            for example in data:
                writer.write(example)
        
        # Calculate file size
        file_size = output_file.stat().st_size
        self.logger.info(f"Saved {len(data)} examples to {output_file} ({file_size} bytes)")
        
        return str(output_file)
    
    def generate_data_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a report about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing dataset statistics
        """
        report = {
            "total_examples": len(df),
            "avg_question_length": df['question'].str.len().mean(),
            "avg_answer_length": df['answer'].str.len().mean(),
            "max_question_length": df['question'].str.len().max(),
            "max_answer_length": df['answer'].str.len().max(),
            "estimated_tokens_per_example": df.apply(
                lambda row: estimate_tokens(f"{row['question']} {row['answer']}"), axis=1
            ).mean(),
            "unique_questions": df['question'].nunique(),
            "duplicate_questions": len(df) - df['question'].nunique()
        }
        
        self.logger.info("Dataset Report:")
        for key, value in report.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.2f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        return report
    
    def process_full_pipeline(self, input_file: str, output_file: str) -> str:
        """
        Run the complete data processing pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output JSONL file
            
        Returns:
            Path to processed training file
        """
        self.logger.info("Starting full data processing pipeline...")
        
        # Load and clean data
        df = self.load_csv_data(input_file)
        df_cleaned = self.clean_data(df)
        
        # Generate report
        self.generate_data_report(df_cleaned)
        
        # Create training format
        training_data = self.create_training_format(df_cleaned)
        
        # Validate data
        is_valid, issues = self.validate_training_data(training_data)
        if not is_valid:
            self.logger.warning(f"Validation issues found: {issues}")
        
        # Save training file
        output_path = self.save_training_file(training_data, output_file)
        
        self.logger.info("Data processing pipeline completed successfully!")
        return output_path