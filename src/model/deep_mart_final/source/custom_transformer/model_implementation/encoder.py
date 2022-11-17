import torch
import torch.nn as nn

import src.model.deep_mart_final.source.custom_transformer.model_implementation.transformer_model_utils as tu
from src.model.deep_mart_final.source.custom_transformer.model_implementation.layers.embedding_layer import Embedding
from src.model.deep_mart_final.source.custom_transformer.model_implementation.layers.encoder_layer import EncoderLayer
from src.model.deep_mart_final.source.custom_transformer.model_implementation.layers.normalisation_layer import Norm
from src.model.deep_mart_final.source.custom_transformer.model_implementation.layers.positional_encoder import PositionalEncoder


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n: int,
        heads: int,
        dropout: float,
        device: torch.device,
    ):
        super(Encoder, self).__init__()
        self.n = n
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, device, dropout=dropout)
        self.layers = tu.get_clones(EncoderLayer(d_model, heads, dropout), n)
        self.norm = Norm(d_model)

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.n):
            x = self.layers[i](x, mask)
        return self.norm(x)
