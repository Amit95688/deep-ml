import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def self_attention(Q, K, V):
    d_k = Q.shape[1]

    scores = (Q @ K.T) / np.sqrt(d_k)

    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))

    attention_weights = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    return attention_weights @ V
