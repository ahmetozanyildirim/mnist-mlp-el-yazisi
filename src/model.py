"""Yapılandırılabilir MLP (çok katmanlı algılayıcı) modeli."""
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


class MLP(nn.Module):
    """784 -> gizli katmanlar -> 10 yapısında tam bağlantılı ağ.

    Her gizli katman: Linear -> (BatchNorm) -> aktivasyon -> (Dropout)
    Son katman her rakam için ham bir skor (logit) verir. Softmax modelin içinde
    değil, loss fonksiyonunun içinde uygulanır (nn.CrossEntropyLoss).
    """

    def __init__(self, hidden=(256, 128), activation="relu", dropout=0.0, batchnorm=False,
                 input_dim=784, num_classes=10):
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
