import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):

    prob = []

    for x in range(len(features)):
        z = sum(features[x][j] * weights[j] for j in range(len(weights))) + bias
        prob.append(1 / (1 + math.exp(-z)))

    mse = sum((labels[i] - prob[i])**2 for i in range(len(labels))) / len(labels)

    return [round(p, 4) for p in prob], round(mse, 4)