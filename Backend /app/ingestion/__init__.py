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
from .metadata_enricher import MetadataEnricher
from .ingestion_manager import IngestionManager

__all__ = ["DataSource", "Preprocessor", "Embedded", "MetadataEnricher", "IngestionManager" ]
