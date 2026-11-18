import math

def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    ex = [math.exp(x - m) for x in scores]
    total = sum(ex)
    return [x / total for x in ex]