"""
data_ingestion_pipeline package

Exports core ingestion components:
- DataSource: handles data loading
- Preprocessor: cleans and normalizes text
- Embedded: embedding utilities (if used)
"""

from .data_load import DataSource
from .preprocessing import Preprocessor
from .embedding import Embedded

__all__ = [
    "DataSource",
    "Preprocessor",
    "Embedded",
]
