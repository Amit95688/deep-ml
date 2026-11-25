import numpy as np

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b):

    input_sequence = np.array(input_sequence)
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)

    for x in input_sequence:
        h = tanh(Wx @ x + Wh @ h + b)

    return h.tolist()