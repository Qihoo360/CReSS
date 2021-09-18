# Cross-model Retrieval between 13C NMR Spectrum and Structure

To test Cross-modal Retrieval between Spectrum and Structure (CReSS) system, please visit : http://cnmr.ai.360.cn/

## Data and model

31,921 molecules with 13C NMR spectra have been collected from [Nmrshiftdb](https://nmrshiftdb.nmr.uni-koeln.de/)，and then were randomly split into training set (25,450) and test set (6,471) by a 4:1 ratio. The reference library here for compound identification task consists of these 31,921 molecules. These two datasets can be downloaded directly from [Google Drive](https://drive.google.com/file/d/1-A1zf2WruvQTtjBaH6TKxHM4eu2YCuLp/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1Q79nGpn-GK-ZVoVUTXm9GA) (password:l5tz). You should decompress it as a folder named data.



NMR encoder and SMILES encoder have been already pretrained on the training set. The model weights can be downloaded directly from [Google Drive](https://drive.google.com/file/d/1TWXVAbsqmUMaDJ-xKmshsO34xWEdzYzt/view) or [Baidu Yun](https://pan.baidu.com/s/1Q79nGpn-GK-ZVoVUTXm9GA) (password:l5tz). You should decompress it as a folder named models.

---------

## Install python packages

```bash
pip install -r requirements.txt
```
---------

## Example 1: Calculating the cosine distance between smiles expressions and 13C NMR spectrum

This example can be seen in example_calculate_cosdistance.py .
```python
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

# The result might be like:
# tensor([[ 0.8309, -0.0687],
#         [-0.1868,  0.8324]])
```

---------

## Example 2: Searching a nmr result in a smiles library

This example can be seen in example_search_library.py .
```python
from example_search_library import *


# Load the model
config_path = "models/2_5_w_model/8.json"
pretrain_model_path = "models/2_5_w_model/8.pth"
model_inference = ModelInference(config_path=config_path,
                                    pretrain_model_path=pretrain_model_path,
                                    device="cpu")

# C(O)C1=CC2(C)CCC3C(C)CCC32C1C
nmr_list = [
    17.7, 20.0, 22.9, 28.9, 29.9, 35.8, 37.6, 39.7, 50.9, 57.3, 61.3, 64.1,
    64.9, 134.0, 146.7
]

# Extract NMR spectral feature vector 
nmr_feature = model_inference.nmr_encode(nmr_list)

# Construct a reference library by extracting structural feature vectors from SMILES strings
# This might take a long time
smiles_feature, smiles_list = get_feature_from_json(
    json_list=["data/val.json", "data/train.json"],
    model_inference=model_inference,
    n=64,
    save_name=None,
    type="smiles",
    flag_get_value=True)

# Get top10 candidates by searching library 
indices, scores = get_topK_result(nmr_feature, smiles_feature, 10)

# Print the result
for (sco, idx) in zip(scores, indices):
    for ii, i in enumerate(idx):
        print("top:", ii, "scores:", sco[ii].item(), "smiles:",
                smiles_list[i])

# The result might be like:
# top: 0 scores: 0.7313205599784851 smiles: C(O)C1=CC2(C)CCC3C(C)CCC32C1C
# top: 1 scores: 0.6523016095161438 smiles: C(C)(C)C1C2CC(CN3C(c4ccccc4)=CCCC23)C2CCCCN21
# top: 2 scores: 0.6491471529006958 smiles: C(=CC(C)O)CCCC1CCC(C)C2CCCN12
# top: 3 scores: 0.6225204467773438 smiles: C(=CCO)(CO)CCCC(C)CCCC(C)CCCC(C)(C)O
# top: 4 scores: 0.622443437576294 smiles: C(O)C1=C2CC(C)(C)CC2CC2(C)C(O)CC12
# top: 5 scores: 0.6000078916549683 smiles: C(=C1CN2CCCC2C(C)(O)C1)C(C)C
# top: 6 scores: 0.5936141014099121 smiles: C1(C)(C)C=C(C)C23C(C)CCC12CCC3O
# top: 7 scores: 0.5924368500709534 smiles: C(C)(C)C1CCC(C)C12CC=C(C)C(O)C2
# top: 8 scores: 0.5919881463050842 smiles: C1(C)CC2CCN(C)CC2C=C1SC
# top: 9 scores: 0.5834836363792419 smiles: C(C)(O)CCC=C1C=C(C)C2CCCCN2C1
```