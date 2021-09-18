import torch
from torch import nn
from model.model import SelfCorrelationBaseModel


class ChemClipModel(SelfCorrelationBaseModel):
    def __init__(self,
                 smiles_model=None,
                 nmr_model=None,
                 frozen_smiles_model=False,
                 frozen_nmr_model=False,
                 flag_use_middleware=False,
                 loss_fn=None,
                 flag_use_big_class=False,
                 class_number=130000,
                 feature_dim=768):
        super(ChemClipModel,
              self).__init__(loss_fn=loss_fn,
                             flag_use_big_class=flag_use_big_class,
                             class_number=class_number)

        self.smiles_model = smiles_model
        self.nmr_model = nmr_model
        self.frozen_smiles_model = frozen_smiles_model
        self.frozen_nmr_model = frozen_nmr_model
        self.flag_use_middleware = flag_use_middleware
        self.feature_dim = feature_dim

        if self.flag_use_middleware:
            self.smiles_middleware = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim))
            self.nmr_middleware = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim))

        if self.frozen_smiles_model:
            self.smiles_model.eval()
        else:
            self.smiles_model.train()
        if self.frozen_nmr_model:
            self.nmr_model.eval()
        else:
            self.nmr_model.train()

    def load_weights(self, path):
        if path is not None:
            model_dict = torch.load(path, map_location=torch.device('cpu'))
            self.load_state_dict(model_dict)

    def smiles_model_eval(self):
        self.frozen_smiles_model = True
        self.smiles_model.eval()

    def smiles_model_train(self):
        self.frozen_smiles_model = False
        self.smiles_model.train()

    def nmr_model_eval(self):
        self.frozen_nmr_model = True
        self.nmr_model.eval()

    def nmr_model_train(self):
        self.frozen_nmr_model = False
        self.nmr_model.train()
