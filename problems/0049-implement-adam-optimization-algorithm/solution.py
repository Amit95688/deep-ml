import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001,
                   beta1=0.9, beta2=0.999,
                   epsilon=1e-8, num_iterations=10):

    m = np.zeros_like(x0, dtype=float)
    v = np.zeros_like(x0, dtype=float)

    x0 = x0.copy()

    for t in range(1, num_iterations + 1):

        g = grad(x0)

        # First moment
        m = beta1 * m + (1 - beta1) * g

        # Second moment
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update parameters
        x0 -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x0