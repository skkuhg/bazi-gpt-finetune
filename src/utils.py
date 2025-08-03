"""
Utility functions for BaZi-GPT fine-tuning project.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


def load_env_vars() -> Dict[str, Any]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dict containing environment configuration
    """
    load_dotenv()
    
    return {
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo-0125'),
        'fine_tuned_model': os.getenv('FINE_TUNED_MODEL_ID'),
        'data_file': os.getenv('DATA_FILE_PATH', 'data/training_data.csv'),
        'output_dir': os.getenv('OUTPUT_DIR', 'output/'),
        'epochs': int(os.getenv('EPOCHS', '3')),
        'batch_size': int(os.getenv('BATCH_SIZE', '1')),
        'learning_rate': float(os.getenv('LEARNING_RATE_MULTIPLIER', '0.3')),
        'test_file': os.getenv('TEST_FILE_PATH', 'tests/test_questions.json'),
        'test_output': os.getenv('TEST_OUTPUT_PATH', 'output/test_results.json'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_file': os.getenv('LOG_FILE', 'logs/fine_tuning.log')
    }


def create_directories(paths: list) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {path}")


def setup_logging(log_file: str = None, log_level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (INFO, DEBUG, etc.)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if log_file is specified
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger('bazi_gpt')
    logger.info(f"Logging initialized. Level: {log_level}")
    
    return logger


def validate_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key format.
    
    Args:
        api_key: OpenAI API key to validate
    
    Returns:
        True if valid format, False otherwise
    """
    if not api_key:
        return False
    
    # OpenAI API keys start with 'sk-' and are typically 51 characters long
    if api_key.startswith('sk-') and len(api_key) >= 20:
        return True
    
    return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


def clean_text(text: str) -> str:
    """
    Clean and normalize text for training data.
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove any control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text.strip()


def truncate_text(text: str, max_length: int = 4000) -> str:
    """
    Truncate text to maximum length while preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find the last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we're not cutting too much
        return text[:last_space] + "..."
    else:
        return text[:max_length-3] + "..."


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count for OpenAI models.
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    # Very rough estimation: ~4 characters per token on average
    return len(text) // 4


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                      suffix: str = '', length: int = 50, fill: str = '█') -> None:
    """
    Print a progress bar to terminal.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix text
        suffix: Suffix text
        length: Character length of bar
        fill: Fill character
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()  # New line on completion