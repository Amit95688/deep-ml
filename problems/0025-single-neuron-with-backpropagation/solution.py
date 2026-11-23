import torch

def train_neuron(features: torch.Tensor,
                 labels: torch.Tensor,
                 initial_weights: torch.Tensor,
                 initial_bias: float,
                 learning_rate: float,
                 epochs: int) -> tuple[list[float], float, list[float]]:

    # Initialize parameters
    weight = initial_weights.clone().detach().reshape(-1)
    bias = torch.tensor(
        initial_bias,
        dtype=features.dtype,
        device=features.device
    )

    labels = labels.reshape(-1)
    m = labels.size(0)

    mse = []

    for epoch in range(epochs):

        # 1. Forward pass
        z = features @ weight + bias
        y_pred = torch.sigmoid(z)

        # 2. Calculate MSE
        loss = torch.mean((labels - y_pred) ** 2)
        mse.append(loss.item())

        # 3. Backpropagation
        dz = 2 * (y_pred - labels) * y_pred * (1 - y_pred)

        weight_grad = (features.T @ dz) / m
        bias_grad = torch.sum(dz) / m

        # 4. Update parameters
        weight = weight - learning_rate * weight_grad
        bias = bias - learning_rate * bias_grad

    return (
        [round(w, 4) for w in weight.tolist()],
        round(bias.item(), 4),
        [round(x, 4) for x in mse]
    )