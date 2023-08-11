from torch import nn
from d2l import torch as d2l
import torch
import math
import copy
from rich.pretty import pprint
import rich
from common_utils.core.common import seed_all

seed_all(42, seed_torch=True)


# _ = nn.MultiheadAttention
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, H, d_model, dropout=0.1, bias=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % H == 0

        self.d_model = d_model  # D
        self.d_k = d_model // H  # stay true to notations
        self.d_q = d_model // H
        self.d_v = d_model // H

        self.H = H  # number of heads

        # shadow my notations
        self.W_q = nn.Linear(self.d_model, self.d_q * self.H, bias=bias) # D x D
        self.W_k = nn.Linear(self.d_model, self.d_k * self.H, bias=bias)
        self.W_v = nn.Linear(self.d_model, self.d_v * self.H, bias=bias)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, embeddings, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches, seq_len, _ = embeddings.size()

        # Apply linear transformations to compute Q, K, V
        # NOTE: here is an important misconception that if you have
        # 8 heads, then you SPLIT the embeddings into 8 parts and
        # then apply linear transformations to each part. This is
        # WRONG. You apply linear transformations to the whole
        # embeddings and then split the result into 8 parts.
        Q = self.W_q(embeddings) # Z @ W_q
        K = self.W_k(embeddings) # Z @ W_k
        V = self.W_v(embeddings) # Z @ W_v
        assert Q.shape == (nbatches, seq_len, self.d_q * self.H)

        # Splitting into multiple heads
        Q_heads = []
        K_heads = []
        V_heads = []
        for head in range(self.H):
            Q_heads.append(Q[:, :, head * self.d_k : (head + 1) * self.d_k])
            K_heads.append(K[:, :, head * self.d_k : (head + 1) * self.d_k])
            V_heads.append(V[:, :, head * self.d_k : (head + 1) * self.d_k])

        # Apply attention to each head
        head_outputs = []
        for q, k, v in zip(Q_heads, K_heads, V_heads):
            x, attn = attention(q, k, v, mask=mask, dropout=self.dropout)
            head_outputs.append(x)
            self.attn = attn  # Store attention

        # Concatenate heads
        x_concat = torch.cat(head_outputs, dim=-1)

        # Apply final linear transformation
        return self.W_o(x_concat)


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
    model = MultiHeadedAttention(H=num_heads, d_model=num_hiddens, bias=False)
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    # check the output shape
    # out_mask = model(X, Y, Y, mask=torch.ones((batch_size, num_queries, num_kvpairs)))
    out_no_mask = model(X, mask=None)
    # torch.save(out_no_mask, "a.pt")
    # pprint(out_no_mask)
    # print(out_no_mask.shape)

    loaded_out = torch.load("a.pt")

    if tensors_are_same(out_no_mask, loaded_out):
        print("The tensors are the same!")
    else:
        print("The tensors are different!")
