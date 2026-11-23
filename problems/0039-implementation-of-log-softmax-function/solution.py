import numpy as np
import math
def log_softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - np.max(scores))
    return np.log(e / np.sum(e))