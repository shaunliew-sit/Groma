"""Annotation converters for HOI datasets to Groma instruction format."""

from .hico_to_instruct import convert_hico_to_grounding
from .swig_to_instruct import convert_swig_to_grounding

__all__ = [
    'convert_hico_to_grounding',
    'convert_swig_to_grounding',
]
