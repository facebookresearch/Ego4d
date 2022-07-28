import torch
from torch.nn import Module


class FeedVideoInput(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, x) -> torch.Tensor:
        return self.model(x["video"])
