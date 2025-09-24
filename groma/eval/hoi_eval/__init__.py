"""HOI Evaluation Package for Groma.

This package provides modular components for Human-Object Interaction (HOI) evaluation
using the Groma multimodal model with HICO-DET and SWIG-HOI datasets.

Main Components:
- HOIExtractor: Extract HOI triplets from model responses using NLP analysis
- HOIVisualizer: Create visual representations of HOI predictions and comparisons
- Dataset utilities: Load and manage HICO-DET and SWIG-HOI datasets
- Evaluation utilities: Calculate HOI detection metrics
- Image processing: Interface with Groma model for image analysis
- Evaluation orchestrator: Coordinate end-to-end HOI evaluation workflows

Usage:
    from groma.eval.hoi_eval import create_orchestrator
    from groma.eval.hoi_eval import POSBasedHOIExtractorNoAdj, HOIVisualizer
"""

# Core classes and functions
from .hoi_extractor import POSBasedHOIExtractorNoAdj
from .visualization import HOIVisualizer
from .evaluation_orchestrator import HOIEvaluationOrchestrator, create_orchestrator

# Utility modules (can be imported as needed)
from . import dataset_utils
from . import evaluation_utils
from . import image_processing

# Category definitions (already exist)
from .hico_categories import HICO_INTERACTIONS, HICO_ACTIONS, HICO_OBJECTS
from .swig_v1_categories import SWIG_INTERACTIONS, SWIG_ACTIONS, SWIG_CATEGORIES

# Evaluators (already exist)
from .hico_evaluator import HICOEvaluator
from .swig_evaluator import SWiGEvaluator

__version__ = "1.0.0"
__author__ = "Groma Team"

# Package-level convenience imports
__all__ = [
    # Core classes
    'POSBasedHOIExtractorNoAdj',
    'HOIVisualizer',
    'HOIEvaluationOrchestrator',
    'create_orchestrator',

    # Evaluators
    'HICOEvaluator',
    'SWiGEvaluator',

    # Category definitions
    'HICO_INTERACTIONS', 'HICO_ACTIONS', 'HICO_OBJECTS',
    'SWIG_INTERACTIONS', 'SWIG_ACTIONS', 'SWIG_CATEGORIES',

    # Utility modules
    'dataset_utils',
    'evaluation_utils',
    'image_processing',
]