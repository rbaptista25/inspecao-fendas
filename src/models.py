from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def bloco(entrada, saida):
            return nn.Sequential(
                nn.Conv2d(entrada, saida, kernel_size=3, padding=1),
                nn.BatchNorm2d(saida),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(bloco(3, 16), bloco(16, 32), bloco(32, 64), bloco(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classificador = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classificador(x)
