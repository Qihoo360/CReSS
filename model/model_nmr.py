from torch import nn
from types import MethodType, FunctionType
from model.model import SelfCorrelationBaseModel


class MlpBlock(nn.Module):
    def __init__(self, channels):
        super(MlpBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, input):
        return input + self.block(input)


class CnnBlock_2(nn.Module):
    def __init__(self, channels):
        super(CnnBlock_2, self).__init__()
        self.layer = nn.Conv1d(1,
                               channels,
                               kernel_size=400,
                               stride=200,
                               padding=200)

    def forward(self, input):
        output_1 = self.layer(input.unsqueeze(1))
        return output_1


def nmr_model_version_8(nmr_input_channels=4000,
                        nmr_output_channels=768,
                        channels=32):
    # assert (nmr_input_channels == 4000,
    #         "nmr_input_channels should be 4000 in this specific model")
    hidden_channels = channels * 21
    return nn.Sequential(
        CnnBlock_2(channels),
        nn.ReLU(),
        nn.Flatten(),
        MlpBlock(hidden_channels),
        nn.ReLU(),
        MlpBlock(hidden_channels),
        nn.ReLU(),
        nn.Linear(hidden_channels, nmr_output_channels),
    )


class NMRModel(SelfCorrelationBaseModel):
    def __init__(self,
                 input_channels=400,
                 nmr_output_channels=768,
                 model_fn=None,
                 **kwargs):
        super(NMRModel, self).__init__(**kwargs)
        self.feature_dim = nmr_output_channels
        if isinstance(model_fn, FunctionType) or isinstance(
                model_fn, MethodType):
            self.model = model_fn(nmr_input_channels=input_channels,
                                  nmr_output_channels=nmr_output_channels)
        elif type(model_fn).__name__ == 'classobj':
            self.model = model_fn
        elif isinstance(model_fn, str):
            self.model = eval(model_fn)(
                nmr_input_channels=input_channels,
                nmr_output_channels=nmr_output_channels)

    def encode(self, input):
        assert (self.model is not None)
        if isinstance(input, tuple) and len(input) == 2:
            input_ids, attention_mask = input
            out_pool = self.model(input_ids, attention_mask)[1]
            return out_pool
        else:
            output = self.model(input)
            return output
