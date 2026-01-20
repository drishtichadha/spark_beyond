"""
Filesystem storage for large objects (DataFrames, models).
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


class FileStorage:
    """
    Handles filesystem storage for PySpark DataFrames and ML models.

    Large objects that cannot be efficiently stored in Redis are persisted
    to the filesystem, with their paths stored in Redis session state.
    """

    def __init__(self, base_path: str = "./data/sessions"):
        """
        Initialize FileStorage.

        Args:
            base_path: Base directory for session files. Defaults to ./data/sessions
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileStorage initialized at {self.base_path.absolute()}")

    def get_session_path(self, session_id: str) -> Path:
        """Get or create the directory for a session."""
        path = self.base_path / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_dataframe(
        self, session_id: str, df: DataFrame, name: str
    ) -> Optional[str]:
        """
        Save a PySpark DataFrame to Parquet format.

        Args:
            session_id: Session identifier
            df: PySpark DataFrame to save
            name: Name for the file (without extension)

        Returns:
            Path to the saved file, or None on failure
        """
        if df is None:
            return None

        path = self.get_session_path(session_id) / f"{name}.parquet"
        temp_path = self.get_session_path(session_id) / f"{name}.tmp.{uuid.uuid4()}"

        try:
            # Write to temp path first for atomicity
            df.write.mode("overwrite").parquet(str(temp_path))

            # Atomic move
            if path.exists():
                shutil.rmtree(path)
            shutil.move(str(temp_path), str(path))

            logger.info(f"Saved DataFrame to {path}")
            return str(path)
        except Exception as e:
            logger.error(f"Failed to save DataFrame {name}: {e}")
            # Cleanup temp path if it exists
            if temp_path.exists():
                shutil.rmtree(temp_path)
            return None

    def load_dataframe(
        self, session_id: str, name: str, spark: SparkSession
    ) -> Optional[DataFrame]:
        """
        Load a PySpark DataFrame from Parquet format.

        Args:
            session_id: Session identifier
            name: Name of the file (without extension)
            spark: SparkSession to use for loading

        Returns:
            Loaded DataFrame, or None if file doesn't exist
        """
        path = self.get_session_path(session_id) / f"{name}.parquet"

        if not path.exists():
            logger.debug(f"DataFrame file not found: {path}")
            return None

        try:
            df = spark.read.parquet(str(path))
            logger.info(f"Loaded DataFrame from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load DataFrame {name}: {e}")
            return None

    def save_model(self, session_id: str, model, name: str = "model") -> Optional[str]:
        """
        Save a Spark ML model.

        Args:
            session_id: Session identifier
            model: Spark ML model with write() method
            name: Name for the model directory

        Returns:
            Path to the saved model, or None on failure
        """
        if model is None:
            return None

        path = self.get_session_path(session_id) / name
        temp_path = self.get_session_path(session_id) / f"{name}.tmp.{uuid.uuid4()}"

        try:
            # Write to temp path first
            model.write().overwrite().save(str(temp_path))

            # Atomic move
            if path.exists():
                shutil.rmtree(path)
            shutil.move(str(temp_path), str(path))

            logger.info(f"Saved model to {path}")
            return str(path)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            if temp_path.exists():
                shutil.rmtree(temp_path)
            return None

    def load_model(
        self, session_id: str, model_class, name: str = "model"
    ) -> Optional[object]:
        """
        Load a Spark ML model.

        Args:
            session_id: Session identifier
            model_class: Class with load() method (e.g., SparkXGBClassifierModel)
            name: Name of the model directory

        Returns:
            Loaded model, or None if not found
        """
        path = self.get_session_path(session_id) / name

        if not path.exists():
            logger.debug(f"Model not found: {path}")
            return None

        try:
            model = model_class.load(str(path))
            logger.info(f"Loaded model from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def delete_session_files(self, session_id: str) -> bool:
        """
        Delete all files for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        path = self.get_session_path(session_id)

        if not path.exists():
            return True

        try:
            shutil.rmtree(path)
            logger.info(f"Deleted session files: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session files {session_id}: {e}")
            return False

    def session_exists(self, session_id: str) -> bool:
        """Check if a session has stored files."""
        path = self.base_path / session_id
        return path.exists() and any(path.iterdir())

    def get_session_size(self, session_id: str) -> int:
        """Get total size of session files in bytes."""
        path = self.base_path / session_id
        if not path.exists():
            return 0

        total = 0
        for file in path.rglob("*"):
            if file.is_file():
                total += file.stat().st_size
        return total

    def cleanup_orphaned_sessions(self, valid_session_ids: set) -> int:
        """
        Remove session directories that are not in the valid set.

        Args:
            valid_session_ids: Set of session IDs that should be kept

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir() and session_dir.name not in valid_session_ids:
                try:
                    shutil.rmtree(session_dir)
                    logger.info(f"Cleaned up orphaned session: {session_dir.name}")
                    cleaned += 1
                except Exception as e:
                    logger.error(f"Failed to cleanup {session_dir.name}: {e}")
        return cleaned
