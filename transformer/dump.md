## Questions

### Last Token in the Input Sequence Is The Key In Causal Masked Models

This means the last token in the input sequence (the last row in [L, D]) is a
function of all previous tokens, so it is not surprising why the tutorial will
just use the last row/token's corresponding prediction as the next predicted
token/word, given all previous tokens.

> Important to know the last token or last row of [L, D] is actually a function
> of all previous tokens, here it is unmasked already. So if confused, just
> remember the pre logits last row, corresponding to the last token in the input
> sequence, is a function of all previous tokens. It just means that row holds
> all information, context, of all previous tokens so we can say its conditioned
> on all previous tokens.

### Why Masked == 0 in some?

The use of `mask == 0` in the `masked_fill` operation is a result of how the
mask is constructed. Essentially, different implementations may represent masks
differently:

1. **Boolean Masking with True/False**: In some implementations, the mask might
   be a Boolean tensor where `True` denotes the positions to mask (set to
   negative infinity) and `False` for the positions to keep. In such cases, you
   can directly use the mask in `masked_fill` as in your provided code:

    ```python
    attention_scores = attention_scores.masked_fill(mask, float("-inf"))
    ```

    Here, if `mask[i][j]` is `True`, `attention_scores[i][j]` would be set to
    `-inf`.

2. **Integer Masking with 1/0**: In other implementations, the mask might be an
   integer tensor where `1` denotes the positions to keep and `0` denotes the
   positions to mask. In such cases, you'll often find the mask is inverted
   (`mask == 0`) before using `masked_fill`:

    ```python
    attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
    ```

    Here, if `mask[i][j]` is `0`, `attention_scores[i][j]` would be set to
    `-inf`.

The core functionality—masking certain positions in the attention scores—is the
same in both cases. The difference lies in how the mask tensor is constructed
and interpreted. So, if you find an implementation using `mask == 0`, it's
likely using an integer mask where `0` signifies positions to mask, whereas if
it's directly using `mask`, it's probably a Boolean mask where `True` signifies
positions to mask.

### Why do we need both ignore index in Loss and also negative infinity mask

Using an "ignore index" in the `CrossEntropyLoss` function in PyTorch can ignore
the effect of certain tokens (like padding tokens) during the loss computation.
However, the purpose of the mask in the attention mechanism and the "ignore
index" in the loss function serve different roles in the model, and they operate
at different stages of the computational graph.

1. **Ignore Index in Loss Function**: The "ignore index" in the loss function
   ensures that the model's output at certain positions (typically corresponding
   to padding tokens) does not contribute to the loss. This happens at the very
   end of the forward pass, just before backpropagation begins.

2. **Mask in Attention Mechanism**: The mask in the attention mechanism, on the
   other hand, operates during the forward pass at the time when attention
   scores are computed. This is a more "internal" operation and ensures that
   certain positions do not contribute to the output at all, not just during the
   loss computation but actually in the intermediate representations (i.e.,
   context vectors) that the model computes.

To put it another way, even if you're ignoring certain tokens in your loss
calculation, those tokens can still influence the model's output unless they're
masked out in the attention mechanism itself.

For example, consider a decoder in a sequence-to-sequence model:

-   If you don't use a mask in the attention mechanism, future tokens could
    influence the output at the current timestep, which is not desirable.
-   Even if you use an "ignore index" in your loss function, it doesn't prevent
    the model from "cheating" by peeking at the future tokens if they are not
    masked in the attention mechanism.

So in summary, using an "ignore index" in `CrossEntropyLoss` is not a
replacement for using attention masks. Both have specific roles in the model,
and they are often used together to ensure both that the model attends to the
right tokens and that it is trained properly.

### Why sometimes unsqueeze masks?

The `unsqueeze` operation is used to add an additional dimension to the tensor.
In attention mechanisms, particularly the scaled dot-product attention used in
models like the Transformer, the masks usually need to have the same number of
dimensions as the attention logits for proper broadcasting.

For instance, let's say your source tensor (`src`) has a shape of $B \times L$
where $B$ is the batch size and $L$ is the sequence length. The attention logit
tensor resulting from the query-key dot product would then have shape
$B \times N \times L \times L$, where $N$ is the number of attention heads.

The mask needs to align with the $L \times L$ dimensions of this 4D tensor. In
order to accomplish that, you add singleton dimensions to make it compatible
with the attention logit tensor. By unsqueezing the mask tensor from
$B \times L$ to $B \times 1 \times 1 \times L$, you enable broadcasting such
that the mask effectively gets expanded to $B \times N \times L \times L$ during
the attention calculation, perfectly aligning with the attention logits.

That's why the line:

```python
self.src_mask = (src != pad).unsqueeze(-2)
```

adds a singleton dimension, converting the shape from $B \times L$ to
$B \times 1 \times 1 \times L$ for proper broadcasting during the attention
computations.

### Am i right to assume that the core idea of autoregressive model like decoder only (GPT like) is that for a given sample, there will eventually be L rows where L is the seq length, and therefore I can intuitively view it as 1 sample having L samples, since for each row, we will compute the loss. Am I right in my understanding? Do not hesistate to correct me.

Your description captures an important aspect of autoregressive models like
decoder-only Transformers (e.g., GPT). Specifically, you're talking about how
the model treats a sequence during training. Let's break down your understanding
step by step.

1. **Sequence Length**: When you mention "L rows," where "L" is the sequence
   length, you're essentially pointing out that each sequence is divided into
   "L" time steps (or tokens). Each time step becomes an input-output pair for
   training the model.

2. **One Sequence as Multiple Samples**: You're correct to intuit that a single
   sequence of length "L" can be treated like "L" samples, at least in the
   context of loss calculation. This is because, during training, the model
   computes the loss at each time step by comparing the predicted token with the
   actual next token in the sequence.

3. **Loss Computation**: The loss is often computed at each position and then
   averaged over the sequence length or summed up, depending on the specific
   loss function or training regime.

However, it's crucial to clarify that although a single sequence may contribute
"L" terms to the loss function, this is not equivalent to having "L" independent
samples. The key difference lies in the autoregressive property: the prediction
at each time step is conditioned on the preceding tokens. This introduces a
temporal dependency across the "L" positions, making them not entirely
independent samples.

In other words, while it's accurate to say that a single sequence contributes
multiple terms to the loss function, these terms are correlated because they
come from the same sequence and are generated in an autoregressive manner.

To summarize, you're mostly correct in your understanding that a single sequence
is broken down into multiple steps for the purpose of loss computation, but it's
important to remember that these steps are not independent samples due to the
autoregressive nature of the model.

### Confusion: Training versus Inference

The statement "it generates one token at a time and uses its own previously
generated tokens as context for generating subsequent tokens" is generally true
for GPT-like models during the inference stage, not during training. During
inference (or generation), the model does indeed use its own previously
generated tokens to produce the next token, since there is no ground truth
sequence to rely on. In that case, if the model makes an incorrect prediction at
a certain timestep, that incorrect token is used as part of the context for the
following timestep.

During training, however, the model typically uses the ground truth tokens for
the preceding sequence as context for predicting each next token, as described
in your example. This resembles teacher forcing, in that the ground truth,
rather than the model's own predictions, is used to guide training.

So there's no contradiction, but the behavior is context-dependent:

-   During training, the ground truth sequence is used for context.
-   During inference, the model's own previously generated tokens are used for
    context.

Both approaches are consistent with the autoregressive nature of the model: in
both cases, the token at each position is generated based on the tokens at all
previous positions. The difference lies in whether those preceding tokens come
from the ground truth (during training) or from the model's own previous outputs
(during inference).

### Training vs Inference

In an autoregressive model like a Transformer decoder, the concept of "learning
the representation of the sequence as it goes" does not refer to the model
processing one token at a time during actual forward passes. Instead, it refers
to the model's ability to generate or predict one token at a time during
inference, while training on a full sequence in a batched manner.

During training:

-   All tokens are processed in parallel for efficiency. This is possible
    because the entire sequence is known beforehand (it's the training data).
-   The "autoregressive" property is enforced by using masks in the
    self-attention mechanism. This masking ensures that the prediction for each
    token can only depend on previously generated tokens, not on future tokens
    which the model has no access to during inference. This is how the model
    learns the conditional probability distribution of each token given the
    previous tokens, despite the parallel processing of tokens.

During inference:

-   The model starts with an initial token (such as a start-of-sequence token)
    and generates the next token based on this single input.
-   Then, the model uses both the initial token and the newly generated token to
    predict the third token, and so on.
-   This process is sequential and each new token is predicted based on the
    previously generated tokens, creating a sequence one token at a time.

So, when we say that the model learns the representation of the sequence as it
goes, we mean that the model is trained to handle sequences in such a way that
it can generate them one piece at a time, respecting the causal order inherent
to the task (e.g., language modeling). The parallel processing during training
does not contradict the autoregressive nature of the model; it is simply a
computational efficiency that is enabled by knowing the full sequence in
advance.

## Some Implementation Details

```
Performs one decoder forward pass given encoder hidden states, the decoder input tokens and attention masks.
B = batch size
S = source sequence length
T = target sequence length
E = embedding dimensionality
V = vocabulary size
```

### Input

Let's view input's first two samples:

```
tensor([[15,  4,  9, 10,  1,  3, 13,  0,  6,  2],
│   │   [15,  3,  5, 10,  4,  6, 13,  0,  8,  1]])
```

which is

-   shape is `[2, 10]` which is `BxL`.
-   `49+13=62` but no `EOS` as we truncated last token.
-   `35+46=81` but no `EOS` as we truncated last token.

### Positional Encodings

#### Why do we hardcode batch size of 1 when creating P?

The tensor $P$ for positional encoding is initialized with a batch size of 1.
This makes it easy to add to the actual input sequences later, during the
forward pass. Positional encodings are not dependent on the specific input
sequence but are a function of the position within the sequence. Therefore, they
can be precomputed and stored. When you look at the forward pass:

```python
def forward(self, Z: torch.Tensor) -> torch.Tensor:
    Z = self._add_positional_encoding(Z)
    return self.dropout(Z)
```

and the `_add_positional_encoding` method:

```python
def _add_positional_encoding(self, Z: torch.Tensor) -> torch.Tensor:
    """Add the positional encoding tensor to the input tensor."""
    return Z + self.P[:, : Z.shape[1], :].to(Z.device)
```

You'll see that $P$ is sliced to match the sequence length of $Z$ and then added
to $Z$. Because of broadcasting rules in PyTorch, $P$ will automatically be
broadcasted to the batch size of $Z$ during this addition. This is why $P$ is
initialized with a batch size of 1; it keeps the implementation flexible while
making the broadcasting implicit.

#### Why do we register P as a buffer in PyTorch?

In your `PositionalEncoding` class, the tensor `self.P` holds the pre-computed
positional encodings. If you intend for this tensor to be automatically moved to
the correct device when the module is moved, and if it should not be a learnable
parameter, then registering it as a buffer would be a good idea. This ensures
that `self.P` is part of the module's state but is not updated during
backpropagation.

You could register `self.P` as a buffer right after you initialize it in the
`_init_positional_encoding` method:

```python
def _init_positional_encoding(self) -> torch.Tensor:
    """Initialize the positional encoding tensor."""
    P = torch.zeros((1, self.max_seq_len, self.d_model))
    position = self._get_position_vector()
    div_term = self._get_div_term_vector()
    P[:, :, 0::2] = torch.sin(position / div_term)
    P[:, :, 1::2] = torch.cos(position / div_term)
    self.register_buffer("P", P, persistent=True)
    return P
```

Using `register_buffer` ensures that:

1. `self.P` is automatically moved to the device the model is moved to (e.g.,
   from CPU to GPU).
2. `self.P` is saved when you save the model using `torch.save` or `torch.load`.

The `persistent=False` argument indicates that the buffer should not be part of
the model's `state_dict`, meaning it won't be saved or loaded with the model. If
you do want it to be part of the `state_dict`, you can simply omit this
argument.

### Attention

#### Why do we call contiguous on Q, K and V?

D2L's code uses `reshape` to reshape the `Q`, `K` and `V`, where other code such
as from the Annotated Transformer uses `view`. When you use `view`, this assumes
the tensor is `contiguous`, so it is better to call `contiguous` first.

#### Why do we want to transpose Q, K, and V?

The transposition of $Q$, $K$, and $V$ in multi-head attention serves a specific
purpose: to allow for parallel computation across multiple attention heads. In
the original shape, the "heads" dimension does not exist; the tensor is simply
$B \times L \times D$, where $B$ is the batch size, $L$ is the sequence length,
and $D$ is the model dimension. By transposing, we create a new shape
$B \times H \times L \times (D/H)$, where $H$ is the number of heads. This
enables the following:

1. **Parallelization**: Each head can now be computed in parallel since each
   head operates independently of the others.
2. **Optimization**: Modern hardware accelerators like GPUs are optimized for
   certain tensor operations, and having a shape that aligns well with these
   optimizations can result in faster computation.
3. **Readability and Maintainability**: It's easier to understand and debug the
   operations for each head when they're isolated like this.

#### Why do we want to reverse transpose Q, K, and V?

After the attention scores are computed and used to weight $V$, we get a new
tensor for each head. However, these tensors are still in the transposed shape
$B \times H \times L \times (D/H)$, and they need to be concatenated and
linearly transformed to continue through the network. The reverse transposition
essentially does the following:

1. **Concatenation**: Converts the multiple heads back into a single tensor.
   This is required because subsequent layers (like feed-forward neural
   networks) expect input in the original $D$-dimensional space.

2. **Compatibility**: The rest of the neural network architecture often expects
   input tensors to have a specific shape (usually $B \times L \times D$).
   Reverse transposing ensures that the output of the multi-head attention block
   can be fed into subsequent layers without issue.

3. **Resource Efficiency**: By reducing the tensor back to its original
   dimensions, we can save memory and computational resources, which is
   beneficial when you're training large models or operating under hardware
   constraints.

In summary, the initial transposition is done to facilitate parallel computation
across heads, and the reverse transposition is done to concatenate these heads
and prepare the tensor for subsequent layers.
