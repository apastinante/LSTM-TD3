"""
Serialization utilities for saving configuration and other data
"""

import numpy as np
import torch
import json

def convert_json(obj):
    """
    Convert an object to a JSON-serializable format.
    Handles numpy arrays, tensors, and other common non-serializable types.
    """
    if isinstance(obj, dict):
        return {k: convert_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_json(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, torch.device):
        return str(obj)
    elif callable(obj):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting their __dict__
        return convert_json(obj.__dict__)
    else:
        return obj

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False