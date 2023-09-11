import copy
import math

import rich
import torch
from d2l import torch as d2l
from rich.pretty import pprint
from torch import nn

from common_utils.core.common import seed_all

seed_all(42, seed_torch=True)


# _ = nn.MultiheadAttention
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, bias=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias)
        self.W_v = nn.Linear(d_model, d_model, bias)
        self.W_o = nn.Linear(d_model, d_model, bias)
        pprint(self.W_q.weight.shape)

        # self.linears = clones(nn.Linear(d_model, d_model, bias), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # NOTE: I do not like to pass q,k,v into forward as what is supposed to be passed
    # is the embeddings.
    def forward(self, embeddings, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches, seq_len, hidden_dim = embeddings.size()
        print(nbatches, seq_len, hidden_dim)

        # embeddings shape = [2, 4, 100] [batch, seq_len, D]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        Q = self.W_q(embeddings).view(nbatches, seq_len, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(embeddings).view(nbatches, seq_len, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(embeddings).view(nbatches, seq_len, self.h, self.d_k).transpose(1, 2)
        pprint(Q.shape)

        # Apply attention
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        # Concatenate heads and apply final linear transformation
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.W_o(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def tensors_are_same(tensor1, tensor2, atol=1e-8, rtol=1e-5):
    return torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)


if __name__ == "__main__":
    num_hiddens, num_heads = 100, 1
    model = MultiHeadedAttention(h=num_heads, d_model=num_hiddens, bias=False)
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    # check the output shape
    # out_mask = model(X, Y, Y, mask=torch.ones((batch_size, num_queries, num_kvpairs)))
    out_no_mask = model(X, mask=None)
    # torch.save(out_no_mask, "double_head_ground_truth_attention.pt")
    # pprint(out_no_mask)
    # print(out_no_mask.shape)

    loaded_out = torch.load("double_head_ground_truth_attention.pt")

    if tensors_are_same(out_no_mask, loaded_out):
        print("The tensors are the same!")
    else:
        print("The tensors are different!")
