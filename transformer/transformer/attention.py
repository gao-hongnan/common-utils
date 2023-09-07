from torch import nn

import torch
import math
import copy
from rich.pretty import pprint
import rich
from common_utils.core.common import seed_all
from d2l import torch as d2l
from typing import Optional
from torch.nn import Transformer
from abc import ABC, abstractmethod
from common_utils.core.common import seed_all

seed_all(42, seed_torch=True)


class Attention(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: nn.Dropout = nn.Dropout(p=0, inplace=False),
    ) -> torch.Tensor:
        ...


class ScaledDotProductAttention(Attention):
    def __init__(self, dropout: nn.Dropout = nn.Dropout(p=0, inplace=False)) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: nn.Dropout = nn.Dropout(p=0, inplace=False),
):
    "Compute 'Scaled Dot Product Attention'"
    # 1. Find the embedding dimension D or d_q from the query feature vector.
    #    Q = Z @ W_q \in R^{L x D}
    #    Q_h = Q @ W_q^h \in R^{L x d_q}
    d_q = query.size(dim=-1)

    # 2. Compute the dot product of the query feature vector with the key feature vector.
    #    Note since key is of dim (batch_size, L, d_k) so we operate the
    #    transpose on the last two dimensions, specified by dim0 and dim1.
    #    key.transpose(dim0=-2, dim1=-1) means let the second last dimension
    #    be the last dimension and let the last dimension be the second last dimension.
    attention_scores = torch.bmm(query, key.transpose(dim0=-2, dim1=-1)) / math.sqrt(
        d_q
    )
    torch.testing.assert_close(
        attention_scores,
        torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / math.sqrt(d_q),
        msg="attention scores from bmm and matmul should be the same.",
    )
    # 3. Apply mask to the scores if mask is not None.
    if mask is not None:
        # TODO: give example of shape of mask
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

    # 4. Apply softmax to the scores.
    attention_weights = attention_scores.softmax(dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    context_vector = torch.bmm(attention_weights, value)
    torch.testing.assert_close(
        context_vector,
        torch.matmul(attention_weights, value),
        msg="context vector from bmm and matmul should be the same.",
    )
    return context_vector, attention_weights


if __name__ == "__main__":
    queries = torch.normal(0, 1, (2, 4, 2))
    keys = torch.normal(0, 1, (2, 4, 2))
    values = torch.normal(0, 1, (2, 4, 2))
    pprint(queries.shape)
    pprint(keys.shape)
    pprint(values.shape)
    valid_lens = None

    attention_ = d2l.DotProductAttention(dropout=0.5)
    attention_.eval()
    attention_out_d2l = attention_(queries, keys, values, valid_lens)
    attention_weights_d2l = attention_.attention_weights
    pprint(attention_weights_d2l)
    pprint(attention_out_d2l)
    print(attention_out_d2l.shape)

    seed_all(42, seed_torch=True)
    attention_out, attention_weights = scaled_dot_product_attention(
        query=queries, key=keys, value=values, mask=None, dropout=None
    )
    pprint(attention_out)
    pprint(attention_weights)
    assert torch.allclose(attention_out_d2l, attention_out)
