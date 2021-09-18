from infer import ModelInference

config_path = "models/2_5_w_model/8.json"
pretrain_model_path = "models/2_5_w_model/8.pth"

model_inference = ModelInference(config_path=config_path,
                                 pretrain_model_path=pretrain_model_path,
                                 device="cpu")

smiles_string = [
    "C(=O)C1(O)C(C=O)=CCC2C(C)(C)CCCC21C", #smiles1
     "C1(C)(C)C=[N+]([O-])C(C)(C)N1C" #smiles2
]

nmr_list = [
        [17.6, 18.3, 22.6, 26.5, 31.7, 33.5, 41.8, 42.0, 
        42.2, 78.34, 140.99, 158.3, 193.4, 203.0], #nmr1
        [23.3, 23.5, 26.1, 60.5, 90.0, 132.1] #nmr2
        ]

smiles_feature = model_inference.smiles_encode(smiles_string)
nmr_feature = model_inference.nmr_encode(nmr_list)
print(model_inference.get_cos_distance(smiles_feature, nmr_feature))