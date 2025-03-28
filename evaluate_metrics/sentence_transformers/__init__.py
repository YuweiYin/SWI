from __future__ import annotations

__version__ = "3.4.1"
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"

import importlib
import os

from .backend import (
    export_dynamic_quantized_onnx_model,
    export_optimized_onnx_model,
    export_static_quantized_openvino_model,
)
from .cross_encoder.CrossEncoder import CrossEncoder
from .datasets import ParallelSentencesDataset, SentencesDataset
from .LoggingHandler import LoggingHandler
from .model_card import SentenceTransformerModelCardData
from .quantization import quantize_embeddings
from .readers import InputExample
from .SentenceTransformer import SentenceTransformer
from .similarity_functions import SimilarityFunction
from .trainer import SentenceTransformerTrainer
from .training_args import SentenceTransformerTrainingArguments

# If codecarbon is installed and the log level is not defined,
# automatically overwrite the default to "error"
if importlib.util.find_spec("codecarbon") and "CODECARBON_LOG_LEVEL" not in os.environ:
    os.environ["CODECARBON_LOG_LEVEL"] = "error"

__all__ = [
    "LoggingHandler",
    "SentencesDataset",
    "ParallelSentencesDataset",
    "SentenceTransformer",
    "SimilarityFunction",
    "InputExample",
    "CrossEncoder",
    "SentenceTransformerTrainer",
    "SentenceTransformerTrainingArguments",
    "SentenceTransformerModelCardData",
    "quantize_embeddings",
    "export_optimized_onnx_model",
    "export_dynamic_quantized_onnx_model",
    "export_static_quantized_openvino_model",
]
