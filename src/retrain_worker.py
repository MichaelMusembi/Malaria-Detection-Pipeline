"""
Background Retraining Worker for Malaria Detection Pipeline

Handles model retraining with new data uploaded via API.
"""

import os
import json
import shutil
import zipfile
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path

from model import train_model, create_model
from preprocessing import create_train_generator, create_validation_generator


class RetrainingWorker:
    """
    Handles background retraining tasks for the malaria detection model.
    """
    
    def __init__(self, models_dir: str = 'models', temp_dir: str = 'temp_training'):
        """
        Initializes the retraining worker.
        
        Args:
            models_dir: Directory to save trained models
            temp_dir: Temporary directory for extracting uploaded data
        """
        self.models_dir = models_dir
        self.temp_dir = temp_dir
        self.retrain_status = {}
        
        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
    
    def extract_training_data(
        self,
        zip_path: str,
        task_id: str
    ) -> Optional[str]:
        """
        Extracts uploaded ZIP file containing training data.
        
        Expected ZIP structure:
        training_data.zip
        ├── Parasitized/
        │   ├── image1.png
        │   └── ...
        └── Uninfected/
            ├── image1.png
            └── ...
        
        Args:
            zip_path: Path to uploaded ZIP file
            task_id: Unique identifier for this retraining task
            
        Returns:
            Path to extracted data directory, or None if extraction failed
        """
        extract_path = os.path.join(self.temp_dir, task_id)
        
        try:
            print(f"📦 Extracting training data from {zip_path}...")
            
            # Remove existing directory if it exists
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            print(f"✅ Data extracted to {extract_path}")
            
            # Validate structure
            if not self._validate_data_structure(extract_path):
                print("❌ Invalid data structure")
                return None
            
            return extract_path
            
        except Exception as e:
            print(f"❌ Failed to extract training data: {e}")
            return None
    
    def _validate_data_structure(self, data_dir: str) -> bool:
        """
        Validates that extracted data has the correct structure.
        
        Args:
            data_dir: Path to extracted data directory
            
        Returns:
            True if structure is valid, False otherwise
        """
        # Look for Parasitized and Uninfected folders
        # They might be in root or in a subdirectory
        
        required_folders = ['Parasitized', 'Uninfected']
        
        # Check root directory
        root_folders = os.listdir(data_dir)
        if all(folder in root_folders for folder in required_folders):
            print("✅ Found required folders in root directory")
            return True
        
        # Check subdirectories (one level deep)
        for subdir in root_folders:
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.isdir(subdir_path):
                subdir_folders = os.listdir(subdir_path)
                if all(folder in subdir_folders for folder in required_folders):
                    print(f"✅ Found required folders in {subdir}")
                    # Move folders up to root
                    for folder in required_folders:
                        src = os.path.join(subdir_path, folder)
                        dst = os.path.join(data_dir, folder)
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.move(src, dst)
                    return True
        
        print(f"❌ Could not find required folders: {required_folders}")
        return False
    
    def retrain_model(
        self,
        training_data_dir: str,
        task_id: str,
        epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, any]:
        """
        Retrains the model with new data.
        
        Args:
            training_data_dir: Path to training data directory
            task_id: Unique identifier for this retraining task
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with retraining results
        """
        try:
            print(f"🚀 Starting model retraining (task: {task_id})...")
            
            # Update status
            self.retrain_status[task_id] = {
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "progress": 0
            }
            
            # Generate new model path with version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_path = os.path.join(
                self.models_dir,
                f"malaria_mobilenet_v{timestamp}.h5"
            )
            
            # Train model
            print(f"📚 Training data directory: {training_data_dir}")
            model, history = train_model(
                train_dir=training_data_dir,
                epochs=epochs,
                batch_size=batch_size,
                model_path=new_model_path,
                fine_tune=False  # Skip fine-tuning for faster retraining
            )
            
            # Calculate final metrics
            final_metrics = {
                "val_accuracy": float(history['val_accuracy'][-1]),
                "val_loss": float(history['val_loss'][-1]),
                "val_precision": float(history['val_precision'][-1]),
                "val_recall": float(history['val_recall'][-1]),
                "val_auc": float(history['val_auc'][-1])
            }
            
            # Update model version info
            version_info = {
                "version": timestamp,
                "trained_date": datetime.now().isoformat(),
                "epochs": epochs,
                "task_id": task_id,
                "model_path": new_model_path,
                "metrics": final_metrics
            }
            
            version_path = os.path.join(self.models_dir, 'model_version.json')
            with open(version_path, 'w') as f:
                json.dump(version_info, f, indent=2)
            
            # Update main model symlink/copy
            main_model_path = os.path.join(self.models_dir, 'malaria_mobilenet.h5')
            if os.path.exists(main_model_path):
                # Backup old model
                backup_path = main_model_path.replace('.h5', '_backup.h5')
                shutil.copy(main_model_path, backup_path)
                print(f"📦 Backed up old model to {backup_path}")
            
            # Copy new model to main path
            shutil.copy(new_model_path, main_model_path)
            print(f"✅ Updated main model at {main_model_path}")
            
            # Update status
            self.retrain_status[task_id] = {
                "status": "completed",
                "started_at": self.retrain_status[task_id]["started_at"],
                "completed_at": datetime.now().isoformat(),
                "progress": 100,
                "model_path": new_model_path,
                "metrics": final_metrics
            }
            
            print(f"✅ Retraining completed successfully!")
            print(f"📊 Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
            
            return self.retrain_status[task_id]
            
        except Exception as e:
            print(f"❌ Retraining failed: {e}")
            
            self.retrain_status[task_id] = {
                "status": "failed",
                "started_at": self.retrain_status.get(task_id, {}).get("started_at"),
                "failed_at": datetime.now().isoformat(),
                "error": str(e)
            }
            
            return self.retrain_status[task_id]
        
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(training_data_dir)
    
    def _cleanup_temp_files(self, training_data_dir: str) -> None:
        """
        Removes temporary training data files.
        
        Args:
            training_data_dir: Path to temporary training data directory
        """
        try:
            if os.path.exists(training_data_dir):
                print(f"🧹 Cleaning up temporary files at {training_data_dir}")
                shutil.rmtree(training_data_dir)
        except Exception as e:
            print(f"⚠️ Failed to cleanup temp files: {e}")
    
    def get_retrain_status(self, task_id: str) -> Optional[Dict[str, any]]:
        """
        Returns the status of a retraining task.
        
        Args:
            task_id: Unique identifier for the retraining task
            
        Returns:
            Status dictionary or None if task not found
        """
        return self.retrain_status.get(task_id)
    
    def list_retrain_tasks(self) -> Dict[str, any]:
        """
        Returns all retraining task statuses.
        
        Returns:
            Dictionary of all task statuses
        """
        return self.retrain_status.copy()


# Global worker instance
_worker = RetrainingWorker()


def start_retraining(
    zip_path: str,
    task_id: str,
    epochs: int = 10,
    batch_size: int = 32
) -> Dict[str, any]:
    """
    Convenience function to start a retraining task.
    
    Args:
        zip_path: Path to uploaded ZIP file with training data
        task_id: Unique identifier for this task
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with retraining results
    """
    # Extract training data
    training_data_dir = _worker.extract_training_data(zip_path, task_id)
    
    if training_data_dir is None:
        return {
            "status": "failed",
            "error": "Failed to extract or validate training data",
            "task_id": task_id
        }
    
    # Start retraining
    result = _worker.retrain_model(
        training_data_dir,
        task_id,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return result


def get_worker() -> RetrainingWorker:
    """
    Returns the global retraining worker instance.
    
    Returns:
        RetrainingWorker singleton
    """
    return _worker


if __name__ == "__main__":
    """
    Test retraining worker
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model retraining')
    parser.add_argument('zip_path', type=str, help='Path to training data ZIP')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Generate task ID
    task_id = f"test_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🔄 Starting retraining test (task: {task_id})")
    result = start_retraining(
        args.zip_path,
        task_id,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n Retraining Results:")
    print(json.dumps(result, indent=2))