import torch
import torch.nn.functional as F

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    x = input_matrix.float().unsqueeze(0).unsqueeze(0)
    k = kernel.float().unsqueeze(0).unsqueeze(0)
    out = F.conv2d(x, k, padding=padding, stride=stride)
    return out.squeeze(0).squeeze(0)
