import json
import torch
import numpy as np
from model.build_model import build_chem_clip_model


class ModelInference(object):
    def __init__(self, config_path, pretrain_model_path, device):
        assert (config_path is not None, "config_path is None")
        assert (pretrain_model_path is not None, "pretrain_model_path is None")

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        with open(config_path, "r") as f:
            self.config_json = json.loads(f.read())

        self.clip_model = build_chem_clip_model(
            **self.config_json["model_config"])
        self.clip_model.load_weights(pretrain_model_path)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()


    def smiles_encode(self, smiles_str):
        with torch.no_grad():
            if isinstance(smiles_str, str):
                encode_dict = self.clip_model.smiles_model.smiles_tokenizer.encode_plus(
                    text=smiles_str,
                    max_length=self.config_json["dataset_config"]
                    ["smiles_maxlen"],
                    padding='max_length',
                    truncation=True)
                smiles_ids = torch.from_numpy(
                    np.array(encode_dict['input_ids'])).to(self.device).view(
                        1, -1)
                smiles_mask = torch.from_numpy(
                    np.array(encode_dict['attention_mask'])).to(
                        self.device).view(1, -1)
                smiles_tensor = self.clip_model.smiles_model.encode(
                    (smiles_ids, smiles_mask))
                smiles_tensor = self.clip_model.norm_feature(smiles_tensor)
                return smiles_tensor
            else:
                smiles_ids = []
                smiles_mask = []
                for i in smiles_str:
                    encode_dict = self.clip_model.smiles_model.smiles_tokenizer.encode_plus(
                        text=i,
                        max_length=self.config_json["dataset_config"]
                        ["smiles_maxlen"],
                        padding='max_length',
                        truncation=True)
                    smiles_ids.append(
                        torch.tensor(encode_dict['input_ids']).to(self.device))
                    smiles_mask.append(
                        torch.tensor(encode_dict['attention_mask']).to(
                            self.device))
                smiles_ids = torch.stack(smiles_ids)
                smiles_mask = torch.stack(smiles_mask)
                smiles_tensor = self.clip_model.smiles_model.encode(
                    (smiles_ids, smiles_mask))
                smiles_tensor = self.clip_model.norm_feature(smiles_tensor)
                return smiles_tensor

    def nmr_encode(self, nmr_list):
        with torch.no_grad():
            if not isinstance(nmr_list[0], list):
                # single nmr
                nmr_tensor = self.nmr2tensor(nmr_list).to(
                    self.device)
                nmr_tensor = self.clip_model.nmr_model.encode(
                    nmr_tensor.view(1, -1))
                nmr_tensor = self.clip_model.norm_feature(nmr_tensor)
                return nmr_tensor
            else:
                # batch nmr
                nmr_tensor = [
                    self.nmr2tensor(i).to(self.device)
                    for i in nmr_list
                ]
                
                nmr_tensor = torch.stack(nmr_tensor)
                nmr_tensor = self.clip_model.nmr_model.encode(nmr_tensor)
                nmr_tensor = self.clip_model.norm_feature(nmr_tensor)
                return nmr_tensor

    def get_cos_distance(self, input_1, input_2):
        with torch.no_grad():
            return input_1 @ input_2.t()

    def nmr2tensor(self, nmr, scale=10, min_value=-50, max_value=350):
        units = (max_value - min_value) * scale
        item = np.zeros(units)
        nmr = [round((value - min_value) * scale) for value in nmr]
        for index in nmr:
            if index < 0:
                item[0] = 1
            elif index >= units:
                item[-1] = 1
            else:
                item[index] = 1
        item = torch.from_numpy(item).to(torch.float32)
        return item


