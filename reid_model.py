# src/elephant_vision/reid_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class ElephantReIDModel(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=None):
        super().__init__()
        backbone = models.resnet50(weights="IMAGENET1K_V2")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

        self.classifier = None
        if num_classes:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.flatten(1)
        x = self.embedding(x)
        x = self.bn(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        if self.classifier:
            logits = self.classifier(x)
            return x, logits
        return x
