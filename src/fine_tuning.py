"""
Fine-tuning workflow for BaZi-GPT model.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI
from .data_processor import BaZiDataProcessor
from .utils import load_env_vars, validate_api_key, setup_logging, format_file_size


class BaZiFineTuner:
    """Handles the complete fine-tuning workflow for BaZi-GPT."""
    
    def __init__(self, api_key: str = None, config_file: str = "config/hyperparameters.json"):
        """
        Initialize the fine-tuner.
        
        Args:
            api_key: OpenAI API key (if not provided, will load from environment)
            config_file: Path to hyperparameters configuration file
        """
        self.logger = setup_logging()
        
        # Load configuration
        if not api_key:
            env_vars = load_env_vars()
            api_key = env_vars['api_key']
        
        if not validate_api_key(api_key):
            raise ValueError("Invalid or missing OpenAI API key")
        
        self.client = OpenAI(api_key=api_key)
        self.config = self._load_config(config_file)
        self.data_processor = BaZiDataProcessor()
        
        self.logger.info("BaZi Fine-tuner initialized successfully")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load fine-tuning configuration from file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_file}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default fine-tuning configuration."""
        return {
            "model": "gpt-3.5-turbo-0125",
            "n_epochs": 3,
            "batch_size": 1,
            "learning_rate_multiplier": 0.3,
            "prompt_loss_weight": 0.01,
            "validation_split": 0.1
        }
    
    def prepare_data(self, data_file: str, output_file: str = "output/training_data.jsonl") -> str:
        """
        Prepare training data for fine-tuning.
        
        Args:
            data_file: Path to input CSV file
            output_file: Path to output JSONL file
            
        Returns:
            Path to prepared training file
        """
        self.logger.info(f"Preparing training data from {data_file}")
        
        # Process data using the data processor
        training_file = self.data_processor.process_full_pipeline(data_file, output_file)
        
        self.logger.info(f"Training data prepared successfully: {training_file}")
        return training_file
    
    def upload_training_file(self, file_path: str) -> str:
        """
        Upload training file to OpenAI.
        
        Args:
            file_path: Path to training file
            
        Returns:
            File ID from OpenAI
        """
        self.logger.info(f"Uploading training file: {file_path}")
        
        file_size = Path(file_path).stat().st_size
        self.logger.info(f"File size: {format_file_size(file_size)}")
        
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            file_id = response.id
            self.logger.info(f"File uploaded successfully. File ID: {file_id}")
            return file_id
            
        except Exception as e:
            self.logger.error(f"Error uploading file: {e}")
            raise
    
    def start_fine_tuning(self, file_id: str = None, training_file: str = None) -> str:
        """
        Start the fine-tuning job.
        
        Args:
            file_id: OpenAI file ID (if already uploaded)
            training_file: Path to training file (will upload if file_id not provided)
            
        Returns:
            Fine-tuning job ID
        """
        if not file_id and not training_file:
            raise ValueError("Either file_id or training_file must be provided")
        
        if not file_id:
            file_id = self.upload_training_file(training_file)
        
        self.logger.info("Starting fine-tuning job...")
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=self.config["model"],
                hyperparameters={
                    "n_epochs": self.config["n_epochs"],
                    "batch_size": self.config["batch_size"],
                    "learning_rate_multiplier": self.config["learning_rate_multiplier"]
                }
            )
            
            job_id = response.id
            self.logger.info(f"Fine-tuning job started. Job ID: {job_id}")
            self.logger.info(f"Model: {self.config['model']}")
            self.logger.info(f"Epochs: {self.config['n_epochs']}")
            self.logger.info(f"Batch size: {self.config['batch_size']}")
            self.logger.info(f"Learning rate multiplier: {self.config['learning_rate_multiplier']}")
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error starting fine-tuning job: {e}")
            raise
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            Job status information
        """
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            
            status_info = {
                "id": response.id,
                "status": response.status,
                "model": response.model,
                "fine_tuned_model": response.fine_tuned_model,
                "created_at": response.created_at,
                "finished_at": response.finished_at,
                "training_file": response.training_file,
                "validation_file": response.validation_file,
                "result_files": response.result_files,
                "trained_tokens": response.trained_tokens
            }
            
            return status_info
            
        except Exception as e:
            self.logger.error(f"Error checking job status: {e}")
            raise
    
    def monitor_job(self, job_id: str, check_interval: int = 60) -> Optional[str]:
        """
        Monitor fine-tuning job until completion.
        
        Args:
            job_id: Fine-tuning job ID
            check_interval: Seconds between status checks
            
        Returns:
            Fine-tuned model ID if successful, None if failed
        """
        self.logger.info(f"Monitoring fine-tuning job: {job_id}")
        self.logger.info(f"Check interval: {check_interval} seconds")
        
        while True:
            status_info = self.check_job_status(job_id)
            status = status_info["status"]
            
            self.logger.info(f"Job status: {status}")
            
            if status == "succeeded":
                model_id = status_info["fine_tuned_model"]
                self.logger.info(f"Fine-tuning completed successfully!")
                self.logger.info(f"Fine-tuned model ID: {model_id}")
                self.logger.info(f"Trained tokens: {status_info.get('trained_tokens', 'N/A')}")
                return model_id
                
            elif status == "failed":
                self.logger.error("Fine-tuning job failed!")
                self.logger.error(f"Job details: {status_info}")
                return None
                
            elif status in ["cancelled", "expired"]:
                self.logger.warning(f"Fine-tuning job {status}")
                return None
                
            elif status in ["running", "queued"]:
                self.logger.info(f"Job is {status}, waiting {check_interval} seconds...")
                time.sleep(check_interval)
                
            else:
                self.logger.warning(f"Unknown status: {status}")
                time.sleep(check_interval)
    
    def list_job_events(self, job_id: str) -> list:
        """
        Get events/logs for a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            List of job events
        """
        try:
            response = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
            events = list(response.data)
            
            self.logger.info(f"Retrieved {len(events)} events for job {job_id}")
            return events
            
        except Exception as e:
            self.logger.error(f"Error retrieving job events: {e}")
            return []
    
    def save_job_info(self, job_id: str, output_file: str = "output/job_info.json") -> None:
        """
        Save job information to file.
        
        Args:
            job_id: Fine-tuning job ID
            output_file: Path to output file
        """
        try:
            status_info = self.check_job_status(job_id)
            events = self.list_job_events(job_id)
            
            job_data = {
                "job_info": status_info,
                "events": [
                    {
                        "created_at": event.created_at,
                        "level": event.level,
                        "message": event.message,
                        "type": getattr(event, 'type', None)
                    }
                    for event in events
                ]
            }
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(job_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Job information saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving job info: {e}")
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            True if successfully cancelled, False otherwise
        """
        try:
            self.client.fine_tuning.jobs.cancel(job_id)
            self.logger.info(f"Fine-tuning job {job_id} cancelled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling job: {e}")
            return False
    
    def full_fine_tuning_workflow(self, data_file: str, job_name: str = None) -> Optional[str]:
        """
        Run the complete fine-tuning workflow.
        
        Args:
            data_file: Path to training data CSV
            job_name: Optional name for the job (used in output files)
            
        Returns:
            Fine-tuned model ID if successful, None if failed
        """
        if not job_name:
            job_name = f"bazi_model_{int(time.time())}"
        
        self.logger.info(f"Starting full fine-tuning workflow: {job_name}")
        
        try:
            # Step 1: Prepare data
            training_file = self.prepare_data(data_file, f"output/{job_name}_training.jsonl")
            
            # Step 2: Start fine-tuning
            job_id = self.start_fine_tuning(training_file=training_file)
            
            # Step 3: Monitor job
            model_id = self.monitor_job(job_id)
            
            # Step 4: Save job information
            self.save_job_info(job_id, f"output/{job_name}_job_info.json")
            
            if model_id:
                self.logger.info(f"Workflow completed successfully! Model ID: {model_id}")
            else:
                self.logger.error("Workflow failed - no model ID returned")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return None