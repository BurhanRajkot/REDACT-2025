"""
BackEnd package for MediGuard AI.
Provides utilities to load the ML model and scale medical inputs.
"""

from .model_loader import load_model, load_medical_ranges
from .scaling_bridge import apply_scaling


# Makes Data a package and exposes decoder if needed.
from .decoder import decode_disease  # optional convenience import
