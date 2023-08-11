# Transformer

- [Transformer](#transformer)
  - [Notations](#notations)
    - [Dimensions and Indexing](#dimensions-and-indexing)
    - [General Notations](#general-notations)
    - [Attention Notations](#attention-notations)
  - [GOTCHA](#gotcha)

## Notations

> Mostly follow Lilian's notations, except for $d_k$, $d_q$ and $d_v$, which we
> will stay consistent with the paper.

### Dimensions and Indexing

Here we list common dimensions and indexing used in the Transformer model.
Dimensions and indexing pertaining to attention will be listed in the
[Attention Notations](#attention-notations) section.

- $D$: embedding dimension. In the paper it is denoted as $d_{\text{model}}$.
    We will use $D$ to stay consistent with the rest of the notes.
  - $d$: index of an element in the embedding vector.
- $L$: sequence length.
  - $i$: index of a token in the sequence.
- $V$: vocabulary size.
  - $j$: index of a word in the vocabulary.

### General Notations

- $\mathcal{V}$: is the set of all words in the vocabulary defined as:

    $$
    \mathcal{V} = \{v_1, v_2, ..., v_V\}
    $$

    where

  - $V$: is the size of the vocabulary, also denoted as $|\mathcal{V}|$.
  - $v_j$: is a unique word in the vocabulary $\mathcal{V}$.
  - $j$: is the index of a word in the vocabulary $\mathcal{V}$.

- $\mathbf{X}$: is the input sequence defined as:

    $$
    \mathbf{X} = (x_1, x_2, ..., x_L)
    $$

    where

  - $L$: is the sequence length.
  - $x_{i}$: is a token at position $i$ in the sequence. Each $x_{i}$ is a
        token represented as an integer from the set ${0, 1, ..., V-1}$.
  - $i$: is the index of a token in the sequence $\mathbf{X}$.

- $O$: one-hot representation of the input sequence $\mathbf{X}$. This is a
    $L \times V$ matrix, where each row represents a token in the sequence and
    each column corresponds to a unique word in the vocabulary $\mathcal{V}$.

    $$
    \begin{aligned}
    O &= \begin{bmatrix} o_{1,1} & o_{1,2} & \cdots & o_{1,V} \\ o_{2,1} & o_{2,2} & \cdots & o_{2,V} \\ \vdots & \vdots & \ddots & \vdots \\ o_{L,1} & o_{L,2} & \cdots & o_{L,V} \end{bmatrix} \in \mathbb{R}^{L \times V} \\
    &= \begin{bmatrix} \text{---} & \mathbf{o}_{1, :} & \text{---} \\ \text{---} & \mathbf{o}_{2, :} & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{o}_{L, :} & \text{---} \end{bmatrix} \in \mathbb{R}^{L \times V}
    \end{aligned}
    $$

    where

  - $L$: is the sequence length.
  - $V$: is the vocabulary size.
  - $o_{i, j}$: is the one-hot encoded element at position $i, j$. For a
        given token $x_i$ at the $i$-th position in the sequence $\mathbf{X}$,
        if $f_{\text{stoi}}(x_i)=j$, then the element at position $j$ in the
        one-hot vector for token $x_i$ is 1, and all other elements are 0.
  - $\mathbf{o}_{i, :}$: is the one-hot encoded vector for the token $x_i$
        at the $i$-th position in the sequence $\mathbf{X}$. This row form is
        more important than column form.

- $\mathbf{E}$: is the embedding matrix defined as:

    $$
    \mathbf{E} = \begin{bmatrix} e_{1,1} & e_{1,2} & \cdots & e_{1,D} \\ e_{2,1} & e_{2,2} & \cdots & e_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ e_{V,1} & e_{V,2} & \cdots & e_{V,D} \end{bmatrix} \in \mathbb{R}^{V \times D}
    $$

    where

  - $V$: is the vocabulary size.
  - $D$: is the embedding dimension.
  - $e_{j, d}$: is the embedding element at position $j, d$. For a word
        $v_j$ in the vocabulary $\mathcal{V}$, the corresponding row in
        $\mathbf{E}$ is the embedding vector for that word.

- $\mathbf{Z}$: is the output tensor of the embedding layer, obtained by
    matrix multiplying $O$ with $\mathbf{E}$, and it is defined as:

    $$
    \mathbf{Z} = O \cdot \mathbf{E}
    $$

    $$
    \begin{aligned}
    \mathbf{Z} &= O \cdot \mathbf{E} \\
    &= \begin{bmatrix} z_{1,1} & z_{1,2} & \cdots & z_{1,D} \\ z_{2,1} & z_{2,2} & \cdots & z_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ z_{L,1} & z_{L,2} & \cdots & z_{L,D} \end{bmatrix} \in \mathbb{R}^{L \times D} \\
    &= \begin{bmatrix} \text{---} & \mathbf{z}_{1,:} & \text{---} \\ \text{---} & \mathbf{z}_{2,:} & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{z}_{L,:} & \text{---} \end{bmatrix} \in \mathbb{R}^{L \times D}
    \end{aligned}
    $$

    where

  - $L$: is the sequence length.
  - $D$: is the embedding dimension.
  - $z_{i, d}$: is the element at position $i, d$ in the tensor
        $\mathbf{Z}$. For a token $x_i$ at the $i$-th position in the sequence,
        $z_{i, :}$ is the $D$ dimensional embedding vector for that token.
  - $\mathbf{z}_{i, :}$: is the $D$ dimensional embedding vector for the
        token $x_i$ at the $i$-th position in the sequence.

            In this context, each token in the sequence is represented by a $D$
            dimensional vector. So, the output tensor $\mathbf{Z}$ captures the
            dense representation of the sequence. Each token in the sequence is
            replaced by its corresponding embedding vector from the embedding matrix
            $\mathbf{E}$.

            As before, the output tensor $\mathbf{Z}$ carries semantic information
            about the tokens in the sequence. The closer two vectors are in this
            embedding space, the more semantically similar they are.

- $\mathbf{P}$: is the positional encoding tensor, created with sinusoidal
    functions of different frequencies:

    Each position $i$ in the sequence has a corresponding positional encoding
    vector $p_{i, :}$ of length $D$ (the same as the embedding dimension). The
    elements of this vector are generated as follows:

    $$
    p_{i, 2i} = \sin\left(\frac{i}{10000^{2i / D}}\right)
    $$

    $$
    p_{i, 2i + 1} = \cos\left(\frac{i}{10000^{2i / D}}\right)
    $$

    for each $i$ such that $2i < D$ and $2i + 1 < D$.

    Thus, the entire tensor $\mathbf{P}$ is defined as:

    $$
    \mathbf{P} = \begin{bmatrix} p_{1,1} & p_{1,2} & \cdots & p_{1,D} \\ p_{2,1} & p_{2,2} & \cdots & p_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ p_{L,1} & p_{L,2} & \cdots & p_{L,D} \end{bmatrix} \in \mathbb{R}^{L \times D}
    $$

    where

  - $L$: is the sequence length.
  - $D$: is the embedding dimension.
  - $p_{i, d}$: is the element at position $i, d$ in the tensor
        $\mathbf{P}$.

- Note that $\mathbf{P}$ is independent of $\mathbf{Z}$, and it's computed
    based on the positional encoding formula used in transformers, which uses
    sinusoidal functions of different frequencies:

- OVERWRITING $\mathbf{Z}$: After computing the positional encoding tensor
    $\mathbf{P}$, we can update our original embeddings tensor $\mathbf{Z}$ to
    include positional information:

    $$
    \mathbf{Z} = \mathbf{Z} + \mathbf{P}
    $$

    This operation adds the positional encodings to the original embeddings,
    giving the final embeddings that are passed to subsequent layers in the
    Transformer model.

- Or consider using $\mathbf{Z}^{'}$?

### Attention Notations

- $H$: Number of attention heads.
  - $h$: Index of the attention head.
- $d_k = D/H$: Dimension of the keys. In the multi-head attention case, this
    would typically be $D/H$ where $D$ is the dimensionality of input embeddings
    and $H$ is the number of attention heads.
- $d_q = D/H$: Dimension of the queries. Also usually set equal to $d_k$.
- $d_v = D/H$: Dimension of the values. Usually set equal to $d_k$.
- $\mathbf{W}_{h}^{q} \in \mathbb{R}^{D \times d_q}$: The query weight matrix
    for the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$
    into query representations for the $h$-th head.
  - Important that this matrix collapses to $\mathbf{W}_{1}^q$ when $H=1$
        and has shape $\mathbb{R}^{D \times D}$.
- $\mathbf{W}_{h}^{k} \in \mathbb{R}^{D \times d_k}$: The key weight matrix
    for the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$
    into key representations for the $h$-th head.
  - Important that this matrix collapses to $\mathbf{W}_{1}^k$ when $H=1$
        and has shape $\mathbb{R}^{D \times D}$ since $d_k = D/H = D/1 = D$.
- $\mathbf{W}_{h}^{v} \in \mathbb{R}^{D \times d_v}$: The value weight matrix
    for the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$
    into value representations for the $h$-th head.
  - Important that this matrix collapses to $\mathbf{W}_{1}^v$ when $H=1$
        and has shape $\mathbb{R}^{D \times D}$.
- $\mathbf{W}^q \in \mathbb{R}^{D \times H \cdot d_q = D \times D}$: The query weight
    matrix for all heads. It is used to transform the embeddings $\mathbf{Z}$
    into query representations.

- $\mathbf{W}^k \in \mathbb{R}^{D \times H \cdot d_k = D \times D}$: The key weight matrix
    for all heads. It is used to transform the embeddings $\mathbf{Z}$ into key
    representations.

- $\mathbf{W}^v \in \mathbb{R}^{D \times H \cdot d_v = D \times D}$: The value weight
    matrix for all heads. It is used to transform the embeddings $\mathbf{Z}$
    into value representations.

- $\mathbf{Q} = \mathbf{Z} \mathbf{W}^q \in \mathbb{R}^{L \times D}$: The
    query matrix. It contains the query representations for all the tokens in
    the sequence. This is the matrix that is used to compute the attention
    scores.

- $\mathbf{K} = \mathbf{Z} \mathbf{W}^k \in \mathbb{R}^{L \times D}$: The key
    matrix. It contains the key representations for all the tokens in the
    sequence. This is the matrix that is used to compute the attention scores.

- $\mathbf{V} = \mathbf{Z} \mathbf{W}^v \in \mathbb{R}^{L \times D}$: The
    value matrix. It contains the value representations for all the tokens in
    the sequence. This is the matrix where we apply the attention scores to
    compute the weighted average of the values.

- $\mathbf{q}_{i} = \mathbf{Q}_{i, :} \in \mathbb{R}^{d}$: The query vector
    for the token at position $i$ in the sequence.
- $\mathbf{k}_{i} = \mathbf{K}_{i, :} \in \mathbb{R}^{d}$: The key vector for
    the token at position $i$ in the sequence.
- $\mathbf{v}_{i} = \mathbf{V}_{i, :} \in \mathbb{R}^{d}$: The value vector
    for the token at position $i$ in the sequence.

---

Here's the notation for the provided text. This specifically describes a
multi-head attention operation:

- $d_q$: Dimension of the query before applying the query weight matrix.
- $d_k$: Dimension of the key before applying the key weight matrix.
- $d_v$: Dimension of the value before applying the value weight matrix.

- $\mathbf{W}_i^{(q)} \in \mathbb{R}^{p_q \times d_q}$: Query weight matrix
    for the $i$-th head, used to transform the query.
- $\mathbf{W}_i^{(k)} \in \mathbb{R}^{p_k \times d_k}$: Key weight matrix for
    the $i$-th head, used to transform the key.
- $\mathbf{W}_i^{(v)} \in \mathbb{R}^{p_v \times d_v}$: Value weight matrix
    for the $i$-th head, used to transform the value.

- $p_q$: Dimension of the query after applying the query weight matrix
    $\mathbf{W}_i^{(q)}$. It's the dimension of the query in the attention head
    space.
- $p_k$: Dimension of the key after applying the key weight matrix
    $\mathbf{W}_i^{(k)}$. It's the dimension of the key in the attention head
    space.
- $p_v$: Dimension of the value after applying the value weight matrix
    $\mathbf{W}_i^{(v)}$. It's the dimension of the value in the attention head
    space.

- $f(\cdot)$: Attention function (such as additive attention or scaled
    dot-product attention).

- $h$: Total number of attention heads.

- $\mathbf{h}_i \in \mathbb{R}^{p_v}$: Output of the $i$-th attention head.

- $\mathbf{W}_o \in \mathbb{R}^{p_o \times h p_v}$: Output weight matrix, used
    to transform the concatenation of all head outputs.

- $p_o$: Dimension of the final output after applying the output weight matrix
    $\mathbf{W}_o$.

Let's break this down:

- $\mathbf{h}_i \in \mathbb{R}^{p_v}$: Output of the $i$-th attention head. It
    is computed as a function $f$ which applies attention (such as additive
    attention or scaled dot-product attention) to the transformed queries, keys
    and values. This function depends on the query $\mathbf{q}$, key
    $\mathbf{k}$, and value $\mathbf{v}$, and the weight matrices
    $\mathbf{W}_i^{(q)}$, $\mathbf{W}_i^{(k)}$, and $\mathbf{W}_i^{(v)}$. The
    dimensions $p_q$, $p_k$, and $p_v$ denote the output dimensions of the
    query, key and value transformations respectively, for the $i$-th head.

- $\mathbf{W}_i^{(q)} \in \mathbb{R}^{p_q \times d_q}$,
    $\mathbf{W}_i^{(k)} \in \mathbb{R}^{p_k \times d_k}$, and
    $\mathbf{W}_i^{(v)} \in \mathbb{R}^{p_v \times d_v}$: The weight matrices
    for the $i$-th attention head. These are used to transform the query, key,
    and value inputs to the dimensions suitable for the attention mechanism.

- $f(\cdot)$: This function represents the attention mechanism (like additive
    attention or scaled dot-product attention). It takes as input the
    transformed query, key, and value vectors and produces the output of the
    attention head.

- $\mathbf{W}_o \in \mathbb{R}^{p_o \times h p_v}$: This is the output weight
    matrix that linearly transforms the concatenation of the outputs from all
    attention heads to produce the final output of the multi-head attention
    mechanism.

- The expression
    $\mathbf{W}_o\left[\begin{array}{c}
\mathbf{h}_1 \\
\vdots \\
\mathbf{h}_h
\end{array}\right] \in \mathbb{R}^{p_o}$
    represents the final output of the multi-head attention layer. It's the
    result of applying the linear transformation defined by $\mathbf{W}_o$ to
    the concatenated outputs of all attention heads.

This notation helps us understand the inner workings of the multi-head attention
mechanism, and it provides a clear path for implementing the multi-head
attention mechanism in a neural network model.

## GOTCHA

The notation \( W^{q}_i \) is used in the paper to denote the weight matrix for the queries (Q) of the \( i \)-th head. However, it's essential to understand how this is implemented in practice.

The entire process can be seen as a two-step operation:

1. **Apply Linear Transformations**: You apply linear transformations to the whole embeddings to create larger matrices for Q, K, V. These matrices have dimensions that account for all heads. In practice, this can be implemented using a single linear layer, such as:

   \[
   Q = \text{{embeddings}} @ \mathbf{W}^q
   \]

   where \( \mathbf{W}^q \) has dimensions \( D \times (h \cdot d_q) \).

2. **Reshape and Split**: After applying the linear transformations, you reshape and split the result into individual heads. The reshaping ensures that the final dimensions are \([N, H, S, d_q]\), where \( N \) is the batch size, \( H \) is the number of heads, \( S \) is the sequence length, and \( d_q \) is the dimension of queries per head.

So, while the paper uses notation like \( W^{q}_i \), this doesn't mean that you directly apply a different linear transformation to different parts of the embeddings. Instead, you apply a single large linear transformation to the whole embeddings and then reshape the result to obtain the individual heads.

In mathematical terms, the overall operation can be seen as:

\[
\begin{align*}
Q_{\text{{all heads}}} & = \text{{embeddings}} @ \mathbf{W}^q \\
Q_{\text{{head i}}} & = Q_{\text{{all heads}}}[:, i \cdot d_q : (i + 1) \cdot d_q]
\end{align*}
\]

Here, \( Q_{\text{{all heads}}} \) is the result of applying the linear transformation, and \( Q_{\text{{head i}}} \) is the portion corresponding to the \( i \)-th head, obtained by slicing along the last dimension.