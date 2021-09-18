import torch
from torch import nn
import numpy as np


class SelfCorrelationBaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(
            SelfCorrelationBaseModel,
            self,
        ).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_fn_1 = nn.CrossEntropyLoss()
        self.loss_fn_2 = nn.CrossEntropyLoss()
        self.model = None

    def encode(self, input):
        assert (self.model is not None)
        output = self.model(input)
        return output

    def norm_feature(self, feature):
        return feature / feature.norm(dim=-1, keepdim=True)

    def forward_features(self, features_1, features_2):
        logits = features_1 @ features_2.t()
        return logits

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters()
                            if p.requires_grad)
        return 'Total parameters: {}, Trainable parameters: {}'.format(
            total_num, trainable_num)

    def load_weights(self, path):
        if path is not None:
            model_dict = torch.load(path, map_location=torch.device('cpu'))
            self.load_state_dict(model_dict)
