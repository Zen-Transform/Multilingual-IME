from torch import nn


class LanguageDetectorModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LanguageDetectorModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        return self.model(x)
