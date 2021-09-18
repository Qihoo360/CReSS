from model.model_clip import *
from model.model_nmr import *
from model.model_smiles import *


def build_chem_clip_model(roberta_model_path,
                          roberta_tokenizer_path,
                          smiles_maxlen=300,
                          vocab_size=55,
                          max_position_embeddings=300,
                          num_attention_heads=12,
                          num_hidden_layers=3,
                          type_vocab_size=1,
                          nmr_input_channels=400,
                          nmr_model_fn=None,
                          frozen_smiles_model=False,
                          frozen_nmr_model=False,
                          flag_use_middleware=False,
                          loss_fn=None,
                          smile_use_tanh=False,
                          flag_use_big_class=False,
                          class_number=130000,
                          feature_dim=768):
    if feature_dim <= 0 or feature_dim is None or not isinstance(feature_dim, int):
        feature_dim = 768
    smiles_model = SmilesModel(
        roberta_model_path=roberta_model_path,
        roberta_tokenizer_path=roberta_tokenizer_path,
        smiles_maxlen=smiles_maxlen,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
        smile_use_tanh=smile_use_tanh,
        feature_dim=feature_dim
    )
    nmr_model = NMRModel(input_channels=nmr_input_channels,
                         nmr_output_channels=smiles_model.smiles_config.hidden_size,
                         model_fn=nmr_model_fn)

    clip_model = ChemClipModel(smiles_model=smiles_model,
                               nmr_model=nmr_model,
                               frozen_smiles_model=frozen_smiles_model,
                               frozen_nmr_model=frozen_nmr_model,
                               flag_use_middleware=flag_use_middleware,
                               loss_fn=loss_fn,
                               flag_use_big_class=flag_use_big_class,
                               class_number=class_number,
                               feature_dim=feature_dim
                               )
    return clip_model
