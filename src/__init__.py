"""
LLMs4OL 2026 — Core Library
============================
端到端本体学习核心模块

Usage:
    from src import Config, load_config, detect_gpu
    from src import load_and_prepare_data, OntologyDataset
    from src import setup_model_and_tokenizer
    from src import postprocess_triples
    from src import ReportCallback, generate_report
"""

from .config import Config, load_config, detect_gpu, find_local_model, SYSTEM_PROMPT, MODEL_RECOMMENDATIONS
from .data import load_and_prepare_data, OntologyDataset
from .model import setup_model_and_tokenizer
from .postprocess import postprocess_triples
from .report import ReportCallback, generate_report

__all__ = [
    "Config", "load_config", "detect_gpu", "find_local_model", "SYSTEM_PROMPT", "MODEL_RECOMMENDATIONS",
    "load_and_prepare_data", "OntologyDataset",
    "setup_model_and_tokenizer",
    "postprocess_triples",
    "ReportCallback", "generate_report",
]
