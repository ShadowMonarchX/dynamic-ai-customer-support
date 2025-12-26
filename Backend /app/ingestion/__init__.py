"""
ingestion package

Handles all data ingestion and preprocessing tasks.
Exports:
- DataSource
- Preprocessor
- Embedded
"""

from .data_load import DataSource
from .preprocessing import Preprocessor
from .embedding import  Embedded

__all__ = ["DataSource", "Preprocessor", "Embedded"]
