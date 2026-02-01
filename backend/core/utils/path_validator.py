"""
Path validation utilities to prevent path traversal attacks.

This module provides secure file path validation to ensure that file operations
are restricted to authorized directories only.
"""
from pathlib import Path
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PathValidationError(ValueError):
    """Raised when a path fails security validation"""
    pass


class PathValidator:
    """
    Validates file paths to prevent path traversal attacks.

    Ensures that all file access is restricted to authorized base directories.
    """

    def __init__(self, allowed_base_dirs: list[str]):
        """
        Initialize path validator with allowed base directories.

        Args:
            allowed_base_dirs: List of absolute paths that are allowed for file access
        """
        self.allowed_base_dirs = [Path(d).resolve() for d in allowed_base_dirs]
        if not self.allowed_base_dirs:
            raise ValueError("At least one allowed base directory must be specified")

        logger.info(f"PathValidator initialized with allowed directories: {self.allowed_base_dirs}")

    def validate(self, file_path: str) -> Path:
        """
        Validate and sanitize a file path to prevent traversal attacks.

        Args:
            file_path: User-provided file path (can be relative or absolute)

        Returns:
            Resolved Path object if validation passes

        Raises:
            PathValidationError: If path is outside allowed directories
            FileNotFoundError: If file doesn't exist or is not a regular file
        """
        try:
            # Resolve the path to an absolute path (resolves .. and symlinks)
            requested_path = Path(file_path).resolve()

            # Check if path is within any allowed base directory
            is_allowed = any(
                str(requested_path).startswith(str(base_dir))
                for base_dir in self.allowed_base_dirs
            )

            if not is_allowed:
                logger.warning(f"Path traversal attempt blocked: {file_path} -> {requested_path}")
                raise PathValidationError(
                    f"Access denied: Path '{file_path}' is outside allowed directories. "
                    f"Allowed directories: {[str(d) for d in self.allowed_base_dirs]}"
                )

            # Ensure file exists and is a regular file (not a directory or special file)
            if not requested_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not requested_path.is_file():
                raise PathValidationError(f"Path is not a regular file: {file_path}")

            logger.debug(f"Path validation passed: {file_path} -> {requested_path}")
            return requested_path

        except (OSError, RuntimeError) as e:
            # Handle symlink loops and other OS-level errors
            logger.error(f"Path validation error for {file_path}: {e}")
            raise PathValidationError(f"Invalid path: {file_path}") from e

    def validate_directory(self, dir_path: str) -> Path:
        """
        Validate a directory path.

        Args:
            dir_path: User-provided directory path

        Returns:
            Resolved Path object if validation passes

        Raises:
            PathValidationError: If path is outside allowed directories or not a directory
        """
        try:
            requested_path = Path(dir_path).resolve()

            is_allowed = any(
                str(requested_path).startswith(str(base_dir))
                for base_dir in self.allowed_base_dirs
            )

            if not is_allowed:
                logger.warning(f"Directory access blocked: {dir_path} -> {requested_path}")
                raise PathValidationError(
                    f"Access denied: Directory '{dir_path}' is outside allowed directories"
                )

            if not requested_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")

            if not requested_path.is_dir():
                raise PathValidationError(f"Path is not a directory: {dir_path}")

            logger.debug(f"Directory validation passed: {dir_path} -> {requested_path}")
            return requested_path

        except (OSError, RuntimeError) as e:
            logger.error(f"Directory validation error for {dir_path}: {e}")
            raise PathValidationError(f"Invalid directory path: {dir_path}") from e


# Default validator instance for the application
# These paths are safe defaults - should be configured via environment variables in production
DEFAULT_ALLOWED_DIRS = [
    os.path.abspath("./data"),  # Application data directory
    os.path.abspath("./backend/data"),  # Backend data directory
]

_default_validator: Optional[PathValidator] = None


def get_default_validator() -> PathValidator:
    """
    Get the default path validator instance.

    Returns:
        PathValidator configured with default allowed directories
    """
    global _default_validator
    if _default_validator is None:
        # Allow environment variable to override defaults
        env_allowed_dirs = os.getenv("ALLOWED_DATA_DIRS")
        if env_allowed_dirs:
            allowed_dirs = [d.strip() for d in env_allowed_dirs.split(",")]
        else:
            allowed_dirs = DEFAULT_ALLOWED_DIRS

        _default_validator = PathValidator(allowed_dirs)

    return _default_validator


def validate_file_path(file_path: str) -> Path:
    """
    Convenience function to validate a file path using the default validator.

    Args:
        file_path: User-provided file path

    Returns:
        Validated Path object

    Raises:
        PathValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    return get_default_validator().validate(file_path)


def validate_directory_path(dir_path: str) -> Path:
    """
    Convenience function to validate a directory path using the default validator.

    Args:
        dir_path: User-provided directory path

    Returns:
        Validated Path object

    Raises:
        PathValidationError: If validation fails
        FileNotFoundError: If directory doesn't exist
    """
    return get_default_validator().validate_directory(dir_path)
