import numpy as np


def keys_to_action(held) -> np.ndarray:
    """Map a set of held key chars to a [linear, angular] action in [-1, 1]."""
    linear = (1.0 if "w" in held else 0.0) - (1.0 if "s" in held else 0.0)
    angular = (1.0 if "d" in held else 0.0) - (1.0 if "a" in held else 0.0)
    return np.array([np.clip(linear, -1.0, 1.0),
                     np.clip(angular, -1.0, 1.0)], dtype=np.float32)
