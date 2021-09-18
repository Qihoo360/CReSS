import os
from torch import nn
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from model.model import SelfCorrelationBaseModel


class SmilesModel(SelfCorrelationBaseModel):
    def __init__(self,
                 roberta_model_path,
                 roberta_tokenizer_path,
                 smiles_maxlen=300,
                 vocab_size=55,
                 max_position_embeddings=300,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 type_vocab_size=1,
                 smile_use_tanh=False,
                 feature_dim=768,
                 **kwargs
                 ):
        super(SmilesModel, self).__init__(**kwargs)
        self.smiles_maxlen = smiles_maxlen
        self.feature_dim = feature_dim
        if roberta_tokenizer_path is not None:
            self.smiles_tokenizer = RobertaTokenizer.from_pretrained(
                roberta_tokenizer_path, max_len=self.smiles_maxlen)

        if roberta_model_path is None or not os.path.exists(roberta_model_path):
            self.smiles_config = RobertaConfig(
                vocab_size=vocab_size,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                type_vocab_size=type_vocab_size,
                hidden_size=self.feature_dim
            )
            self.smiles_model = RobertaModel(config=self.smiles_config)
        else:
            self.smiles_config = RobertaConfig.from_pretrained(
                roberta_model_path)
            self.smiles_model = RobertaModel.from_pretrained(
                roberta_model_path)
        self.model = self.smiles_model
        self.smile_use_tanh = smile_use_tanh
        if not self.smile_use_tanh:
            self.dense = nn.Linear(self.feature_dim, self.feature_dim)

    def encode(self, input):
        input_ids, attention_mask = input

        if not self.smile_use_tanh:
            hidden_states = self.model(input_ids, attention_mask)[0]
            first_token_tensor = hidden_states[:, 0]
            pooled_output = self.dense(first_token_tensor)
            return pooled_output

        else:
            # output after pooling
            out_pool = self.model(input_ids, attention_mask)[1]
            return out_pool
