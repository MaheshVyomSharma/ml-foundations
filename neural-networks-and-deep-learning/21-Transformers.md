# 21. Transformers

## 1.1 Why Transformers?

RNNs, LSTMs, and GRUs process sequences recurrently.

They carry information step by step:

```text
x₁ → h₁ → h₂ → h₃ → ... → hₜ
```

This gives them sequence awareness, but also creates limitations:

- computation is inherently sequential
- long-range dependencies require long paths
- gradient flow becomes difficult across very long sequences
- training is harder to parallelize

Attention introduced direct interactions between sequence positions.

The Transformer took the next major step:

```text
Remove Recurrence
      ↓
Use Attention as the Main
Sequence Interaction Mechanism
```

A transformer is therefore a neural-network architecture built primarily around:

```text
Self-Attention
+
Feed-Forward Networks
+
Residual Connections
+
Normalization
+
Positional Information
```

---

## 1.2 The Core Transformer Idea

Suppose the input sequence is:

```text
The animal crossed the road
```

Rather than processing:

```text
The
 ↓
animal
 ↓
crossed
 ↓
...
```

strictly one step at a time, a transformer allows the sequence positions to interact directly through self-attention.

Conceptually:

```text
The ─────────┐
animal ──────┤
crossed ─────┼→ Self-Attention
the ─────────┤
road ────────┘
                ↓
        Contextual Representations
```

Each position can learn which other positions are relevant.

---

## 1.3 Transformer Input

For text, the input normally begins with **tokens**.

```text
Sentence
   ↓
Tokenization
   ↓
Token IDs
   ↓
Embeddings
```

Each token is converted into a vector:

```math
x_i
\in
\mathbb{R}^{d_{\text{model}}}
```

where:

```math
d_{\text{model}}
```

is the model's representation dimension.

These embeddings form the initial numerical representation of the sequence.

---

## 1.4 Embeddings Alone Do Not Represent Order

Consider:

```text
dog bites man
```

and:

```text
man bites dog
```

The same three token embeddings appear in both sequences.

Self-attention alone does not inherently know which token came first.

Therefore, transformers need **positional information**.

Conceptually:

```text
Token Meaning
+
Token Position
      ↓
Transformer Input
```

---

## 1.5 Positional Encoding

A positional representation is added to each token embedding.

Conceptually:

```math
\text{Input Representation}
=
\text{Token Embedding}
+
\text{Position Representation}
```

Thus:

```text
"dog" at position 1
```

is represented differently from:

```text
"dog" at position 5
```

even though the underlying token embedding is the same.

---

## 1.6 Sinusoidal Positional Encoding

The original Transformer introduced deterministic sinusoidal positional encodings.

A simplified form is:

```math
PE_{(pos,2i)}
=
\sin
\left(
\frac{pos}
{10000^{2i/d_{\text{model}}}}
\right)
```

and:

```math
PE_{(pos,2i+1)}
=
\cos
\left(
\frac{pos}
{10000^{2i/d_{\text{model}}}}
\right)
```

Different dimensions therefore oscillate at different frequencies.

This creates a unique positional pattern for every sequence location.

The exact formula is less important than the idea:

```text
Transformer Needs
Explicit Position Information
Because Attention Alone
Does Not Encode Order
```

---

## 1.7 Learned Positional Embeddings

Positional information does not have to be sinusoidal.

Models can also learn position vectors:

```math
p_1,p_2,\ldots,p_n
```

during training.

Then:

```math
z_i
=
x_i+p_i
```

where:

```math
x_i
```

is the token embedding and:

```math
p_i
```

is the position embedding.

Modern architectures use several different positional strategies.

---

## 1.8 Relative Position Information

Instead of representing only:

```text
"This token is at position 10"
```

some architectures focus on relationships such as:

```text
"This token is 3 positions before that token"
```

This is called **relative positional information**.

For many sequence tasks, relative distance can be more useful than absolute position.

Several modern transformers use such mechanisms.

---

## 1.9 Self-Attention Inside a Transformer

Each input representation produces:

```text
Query
Key
Value
```

through learned projections:

```math
Q=XW_Q
```

```math
K=XW_K
```

```math
V=XW_V
```

Self-attention then computes:

```math
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}
{\sqrt{d_k}}
\right)V
```

This produces contextualized representations for every sequence position.

---

## 1.10 Multi-Head Attention

Transformers use several attention heads in parallel.

For each head:

```math
\text{head}_i
=
\text{Attention}
\left(
QW_i^Q,
KW_i^K,
VW_i^V
\right)
```

The heads are then combined:

```math
\text{MultiHead}(Q,K,V)
=
\text{Concat}
\left(
\text{head}_1,\ldots,\text{head}_h
\right)
W^O
```

This allows the model to learn different relationship patterns simultaneously.

Conceptually:

```text
One Sequence
   ↓
Head 1 → Relationship Pattern A
Head 2 → Relationship Pattern B
Head 3 → Relationship Pattern C
...
   ↓
Combine
   ↓
Richer Representation
```

---

## 1.11 Attention Is Only Part of a Transformer Block

A transformer block is not simply:

```text
Self-Attention
```

It normally also contains a **feed-forward neural network**.

A simplified block is:

```text
Input
 ↓
Multi-Head Self-Attention
 ↓
Residual Connection
 ↓
Normalization
 ↓
Feed-Forward Network
 ↓
Residual Connection
 ↓
Normalization
 ↓
Output
```

The exact ordering of normalization differs across transformer variants.

---

## 1.12 Position-Wise Feed-Forward Network

After attention, each sequence position is passed through the same feed-forward network.

A common form is:

```math
FFN(x)
=
W_2
f(W_1x+b_1)
+
b_2
```

where:

```math
f
```

is a non-linear activation function.

The feed-forward network is applied independently to each position.

Thus:

```text
Attention
→ Mix Information Across Positions

Feed-Forward Network
→ Transform Each Position's Features
```

This distinction is central.

---

## 1.13 Attention Mixes Tokens, FFN Mixes Features

A useful mental model is:

```text
Self-Attention
→ "Which other tokens matter to me?"

Feed-Forward Network
→ "How should I transform
   the information I now contain?"
```

Attention primarily mixes information across sequence positions.

The feed-forward network transforms the internal feature representation at each position.

Together they form the core computation of a transformer block.

---

## 1.14 Residual Connections

Transformers use residual or skip connections.

If a sublayer computes:

```math
F(x)
```

the residual form is:

```math
y
=
x+F(x)
```

Conceptually:

```text
x ───────────────┐
│                ↓
│            Addition
│                ↓
└→ Sublayer → F(x)
```

Residual connections provide direct pathways for:

- information flow
- gradient flow

They make deep transformer stacks easier to train.

---

## 1.15 Layer Normalization

Transformers commonly use **Layer Normalization**.

LayerNorm normalizes the features of each individual representation.

Conceptually:

```text
Token Representation
        ↓
Normalize Its Feature Values
        ↓
Scale and Shift
        ↓
Stable Representation
```

LayerNorm is especially suitable because transformer training does not need to rely on statistics across a mini-batch.

---

## 1.16 Residual + LayerNorm

A transformer sublayer commonly appears conceptually as:

```text
Input x
 ↓
Sublayer F(x)
 ↓
Add x
 ↓
LayerNorm
```

or, in many modern architectures:

```text
Input
 ↓
LayerNorm
 ↓
Sublayer
 ↓
Add Original Input
```

These are often called:

```text
Post-Norm
```

and:

```text
Pre-Norm
```

variants.

The exact ordering differs, but both combine normalization with residual pathways.

---

## 1.17 Transformer Encoder

The original Transformer architecture contains an **encoder** and a **decoder**.

The encoder processes the input sequence.

A simplified encoder layer is:

```text
Input Representations
        ↓
Multi-Head Self-Attention
        ↓
Residual + LayerNorm
        ↓
Feed-Forward Network
        ↓
Residual + LayerNorm
        ↓
Encoder Output
```

Several encoder layers can be stacked.

---

## 1.18 Stacked Encoder Layers

A transformer encoder typically uses multiple layers:

```text
Input
 ↓
Encoder Layer 1
 ↓
Encoder Layer 2
 ↓
Encoder Layer 3
 ↓
...
 ↓
Final Contextual Representations
```

Each layer refines the sequence representation.

Lower layers may capture simpler patterns.

Higher layers can represent more complex relationships.

This continues the hierarchical representation-learning idea seen throughout deep learning.

---

## 1.19 Encoder Self-Attention

In the encoder, each token can generally attend to all input positions.

For example:

```text
Token 1 ↔ Token 2
Token 1 ↔ Token 3
Token 1 ↔ Token 4
...
```

There is usually no causal restriction because the complete input sequence is already available.

Thus encoder self-attention is commonly:

```text
Bidirectional Attention
```

in the sense that a token can use both earlier and later input context.

---

## 1.20 Transformer Decoder

The decoder generates an output sequence.

A simplified decoder block contains:

```text
Masked Self-Attention
        ↓
Residual + LayerNorm
        ↓
Cross-Attention to Encoder Output
        ↓
Residual + LayerNorm
        ↓
Feed-Forward Network
        ↓
Residual + LayerNorm
```

The decoder therefore uses two forms of attention:

```text
Self-Attention
+
Cross-Attention
```

---

## 1.21 Masked Self-Attention in the Decoder

When generating a sequence autoregressively, the decoder must not see future output tokens.

Suppose the correct sequence is:

```text
A B C D
```

when predicting:

```text
C
```

the model may see:

```text
A B
```

but not:

```text
D
```

Therefore causal masking is used.

```text
Past
→ Visible

Future
→ Hidden
```

This prevents information leakage during training.

---

## 1.22 Cross-Attention in the Decoder

The decoder also needs information from the encoded input.

Therefore:

```text
Decoder Representations
→ Queries

Encoder Outputs
→ Keys and Values
```

Cross-attention computes:

```text
What part of the input
is relevant to the output
I am currently generating?
```

This connects the input and output sequences.

---

## 1.23 Encoder-Decoder Transformer

The complete original architecture can be summarized as:

```text
Input Sequence
      ↓
Embeddings + Position
      ↓
Transformer Encoder
      ↓
Encoder Representations
      ↓
Cross-Attention
      ↑
Transformer Decoder
      ↑
Previous Output Tokens
      ↓
Next Output Token
```

This is especially natural for tasks such as translation:

```text
Input Sequence
→ Different Output Sequence
```

---

## 1.24 Encoder-Only Transformers

Not every transformer uses both encoder and decoder.

An **encoder-only** transformer uses:

```text
Transformer Encoder Stack
```

without an autoregressive decoder.

These models are well suited to tasks requiring understanding of a complete input.

Examples conceptually include:

```text
Text Classification
Token Classification
Representation Learning
Question Understanding
```

The model can use bidirectional context across the input.

---

## 1.25 Decoder-Only Transformers

A **decoder-only** transformer primarily uses masked self-attention.

Conceptually:

```text
Previous Tokens
      ↓
Masked Self-Attention
      ↓
Transformer Blocks
      ↓
Predict Next Token
```

This architecture is especially suited to **autoregressive language modelling**.

It predicts:

```math
P(x_t
\mid
x_1,x_2,\ldots,x_{t-1})
```

The next token depends only on previous tokens.

Many modern generative language models use this broad architecture.

---

## 1.26 Encoder-Decoder Transformers

Encoder-decoder transformers retain both major components.

They are particularly natural when:

```text
Input Sequence
and
Output Sequence
```

serve different roles.

Examples include:

```text
Translation
Summarization
Sequence Transformation
```

The encoder represents the input.

The decoder generates the output while attending to those representations.

---

## 1.27 The Three Broad Transformer Families

A useful summary is:

| Architecture | Main Strength |
|---|---|
| Encoder-only | Understanding / representation |
| Decoder-only | Autoregressive generation |
| Encoder-decoder | Sequence-to-sequence transformation |

This is a conceptual classification rather than an absolute limitation.

---

## 1.28 Autoregressive Generation

A decoder-only language model generates text one token at a time.

Suppose the prompt is:

```text
Machine learning is
```

The model predicts:

```text
useful
```

The sequence becomes:

```text
Machine learning is useful
```

The model then predicts the next token.

Thus:

```text
Prompt
 ↓
Predict Token
 ↓
Append Token
 ↓
Predict Next Token
 ↓
Repeat
```

This is **autoregressive generation**.

---

## 1.29 Next-Token Prediction

During language-model training, the model may learn to predict each next token.

For a sequence:

```text
Deep learning is powerful
```

training pairs conceptually become:

```text
Deep
→ learning

Deep learning
→ is

Deep learning is
→ powerful
```

The objective encourages the model to estimate:

```math
P(x_t
\mid
x_{<t})
```

where:

```math
x_{<t}
```

means all previous tokens.

---

## 1.30 Softmax Vocabulary Output

Suppose the vocabulary contains:

```math
V
```

possible tokens.

The final model produces logits:

```math
z_1,z_2,\ldots,z_V
```

Softmax converts them into probabilities:

```math
P(x_t=k)
=
\frac{e^{z_k}}
{\sum_{j=1}^{V}e^{z_j}}
```

The model therefore predicts a probability distribution over possible next tokens.

---

## 1.31 Language Model Loss

Next-token prediction commonly uses cross-entropy loss.

If the correct next token is:

```text
learning
```

the model is rewarded for assigning that token high probability.

Conceptually:

```text
Context
 ↓
Next-Token Distribution
 ↓
Compare With Correct Token
 ↓
Cross-Entropy Loss
 ↓
Backpropagation
```

The learning mechanism is exactly the same general neural-network machinery covered earlier in the handbook.

---

## 1.32 Transformers Are Still Neural Networks

Despite their complexity, transformers still contain familiar ingredients.

```text
Linear Layers
Activation Functions
Softmax
Normalization
Residual Connections
Loss Functions
Backpropagation
Optimizers
Regularization
```

The novelty lies primarily in how sequence information is structured and exchanged.

A transformer is not a completely different species of model.

It is an architecture built from neural-network components.

---

## 1.33 Transformer Parameters

Trainable parameters include:

```text
Token Embeddings

Query Projection Weights

Key Projection Weights

Value Projection Weights

Attention Output Projections

Feed-Forward Weights

Normalization Parameters

Final Output Weights
```

All of these are learned using:

```text
Forward Propagation
 ↓
Loss
 ↓
Backpropagation
 ↓
Optimizer
```

The same training loop survives.

---

## 1.34 Why Transformers Parallelize Better Than RNNs

An RNN requires:

```math
h_t
```

before calculating:

```math
h_{t+1}
```

Thus:

```text
Step 1
→ Step 2
→ Step 3
→ Step 4
```

cannot be fully parallelized.

A transformer can compute attention across many positions simultaneously during training.

Conceptually:

```text
All Sequence Positions
        ↓
Parallel Matrix Operations
        ↓
Attention
```

This allows much more efficient use of modern accelerators.

---

## 1.35 Why Parallelism Matters

Deep learning training relies heavily on GPUs and other accelerators.

These devices are extremely effective at:

```text
Large Matrix Multiplications
+
Parallel Numerical Operations
```

Transformers map naturally onto this hardware.

This made it practical to train sequence models at scales that were much harder to achieve with recurrence.

---

## 1.36 Long-Range Dependency Advantage

Suppose two related tokens are 500 positions apart.

In an RNN, information may need to travel through hundreds of recurrent transformations.

In self-attention:

```text
Token A
────────────→ Token B
```

the relationship can be represented directly.

This short path is one major reason transformers handle long-range relationships effectively.

---

## 1.37 Transformer Limitation: Attention Cost

The attention score matrix for sequence length:

```math
n
```

contains approximately:

```math
n^2
```

relationships.

Thus standard self-attention has complexity:

```math
O(n^2)
```

with respect to sequence length.

If sequence length doubles:

```math
n \rightarrow 2n
```

the number of pairwise relationships grows approximately fourfold:

```math
n^2
\rightarrow
4n^2
```

This is a major limitation for very long contexts.

---

## 1.38 Context Window

A transformer's **context window** is the amount of sequence information it can process within one model invocation.

Conceptually:

```text
Tokens Inside Context
→ Available to Attention

Tokens Outside Context
→ Not Directly Available
```

Longer context windows provide access to more information but increase memory and computational requirements.

---

## 1.39 Attention Does Not Mean Unlimited Memory

Transformers are sometimes loosely described as having access to an entire sequence.

In practice, they operate within a finite context window.

Therefore:

```text
Transformer
≠ Infinite Memory
```

Even within the context, information is represented through finite-dimensional learned computations.

Long-context modelling remains an active engineering challenge.

---

## 1.40 Model Dimension

A major transformer hyperparameter is:

```math
d_{\text{model}}
```

the width of token representations.

For example:

```math
d_{\text{model}}=512
```

means each sequence position is represented by a 512-dimensional vector inside major parts of the model.

Larger dimensions increase representational capacity but also increase computation and parameter count.

---

## 1.41 Number of Attention Heads

The number of attention heads:

```math
h
```

is another hyperparameter.

For example:

```text
8 Attention Heads
```

means the attention mechanism operates in eight learned representation subspaces.

More heads do not automatically mean better performance.

The number of heads must work with:

- model dimension
- architecture
- compute budget

---

## 1.42 Feed-Forward Dimension

Transformer blocks commonly expand the representation inside the feed-forward network.

For example:

```text
Model Dimension
512
 ↓
Feed-Forward Hidden Dimension
2048
 ↓
Back to
512
```

Conceptually:

```math
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}
```

This gives each position additional nonlinear transformation capacity.

---

## 1.43 Number of Transformer Layers

Transformers are built by stacking blocks.

```text
Embedding
 ↓
Block 1
 ↓
Block 2
 ↓
Block 3
 ↓
...
 ↓
Block L
```

Increasing:

```math
L
```

increases depth.

Like other deep networks, additional depth can improve representational power but also increases:

- training cost
- memory usage
- optimization difficulty
- parameter count

---

## 1.44 Transformer Scaling

Transformer capacity broadly increases through dimensions such as:

```text
Depth
→ More Layers

Width
→ Larger Representations

Feed-Forward Size
→ Greater Per-Token Capacity

Attention Heads
→ More Attention Subspaces
```

Large language models scale several of these dimensions substantially.

However, simply making a model larger does not remove the need for:

- sufficient data
- stable optimization
- appropriate regularization
- adequate compute

---

## 1.45 Residual Connections Revisited

A transformer may contain dozens or hundreds of layers.

Without residual paths, gradients would have to travel through every transformation sequentially.

Residual connections create:

```math
y=x+F(x)
```

which provides a direct identity path.

This connects directly to the earlier discussion of vanishing gradients.

The same fundamental training problem reappears, and the same architectural idea helps solve it.

---

## 1.46 LayerNorm Revisited

LayerNorm appears throughout transformers because it helps keep representations numerically stable.

Recall:

```text
BatchNorm
→ Depends on Batch Statistics

LayerNorm
→ Normalizes Features
  Within Each Example
```

Sequence-model training often benefits from this independence from batch-level statistics.

This is why LayerNorm became strongly associated with transformer architectures.

---

## 1.47 Dropout in Transformers

Dropout may be applied to:

- attention outputs
- feed-forward layers
- embeddings
- residual pathways

Conceptually:

```text
Large Transformer Capacity
        ↓
Overfitting Risk
        ↓
Dropout / Other Regularization
```

The same regularization principles from ordinary neural networks remain applicable.

---

## 1.48 Training a Transformer

The training loop remains:

```text
Token Batch
    ↓
Embeddings
    ↓
Transformer Blocks
    ↓
Predictions
    ↓
Loss
    ↓
Backpropagation
    ↓
Optimizer
    ↓
Updated Parameters
```

The internal architecture changed.

The learning principle did not.

---

## 1.49 Transformer Optimization

Transformers are often trained using optimizers such as:

```text
Adam
AdamW
```

along with carefully designed:

```text
Learning Rate Schedules
Warm-Up
Weight Decay
Gradient Clipping
```

Large models are particularly sensitive to optimization settings.

Thus all the earlier chapters on:

- initialization
- gradients
- optimizers
- regularization
- normalization

remain directly relevant.

---

## 1.50 Learning Rate Warm-Up

Large transformer models may begin training with a very small learning rate and gradually increase it.

Conceptually:

```text
Training Start
→ Tiny Learning Rate

Early Steps
→ Gradually Increase

Stable Training Established
→ Follow Normal Schedule
```

This is called **learning-rate warm-up**.

It can reduce instability during the earliest optimization steps.

---

## 1.51 Pretraining

A transformer can be trained on a very large general dataset before being adapted to a specific task.

This first stage is called **pretraining**.

Conceptually:

```text
Large General Dataset
       ↓
Pretraining
       ↓
General Representations
       ↓
Adapt to Specific Task
```

This follows the same broad transfer-learning principle seen with CNNs.

---

## 1.52 Fine-Tuning

After pretraining, the model can be trained further on a smaller task-specific dataset.

This is **fine-tuning**.

```text
Pretrained Transformer
        ↓
Task-Specific Data
        ↓
Continue Training
        ↓
Specialized Model
```

Fine-tuning modifies pretrained parameters rather than learning the entire model from scratch.

---

## 1.53 Feature Extraction With Transformers

A pretrained transformer can also be used as a representation generator.

Conceptually:

```text
Input
 ↓
Pretrained Transformer
 ↓
Contextual Embeddings
 ↓
Separate Classifier / Model
```

The transformer may remain partly or completely frozen.

This is analogous to using a pretrained CNN as a feature extractor.

---

## 1.54 Foundation Models

When very large models are pretrained on broad datasets and later adapted to many downstream tasks, they are often called **foundation models**.

The central idea is:

```text
Train General Capabilities Once
        ↓
Reuse Across Many Tasks
```

Large language models are prominent examples of this broader paradigm.

---

## 1.55 What Is a Language Model?

A **language model** estimates probabilities over sequences of tokens.

For an autoregressive model:

```math
P(x_1,x_2,\ldots,x_n)
=
\prod_{t=1}^{n}
P(x_t\mid x_1,\ldots,x_{t-1})
```

In words:

```text
Probability of Whole Sequence
=
Product of
Next-Token Probabilities
```

A transformer can implement these conditional probability estimates at large scale.

---

## 1.56 From Transformer to LLM

A **Large Language Model (LLM)** is not a fundamentally new mathematical mechanism beyond transformers.

Broadly:

```text
Transformer Architecture
+
Very Large Parameter Count
+
Very Large Training Dataset
+
Large Compute
+
Language Modelling Objective
+
Additional Training / Alignment
```

produces the class of systems commonly called large language models.

Thus the path is:

```text
Neuron
 ↓
Neural Network
 ↓
Deep Network
 ↓
Attention
 ↓
Transformer
 ↓
Large Transformer
 ↓
Language Model
```

There is continuity throughout.

---

## 1.57 Small Language Models

A **Small Language Model (SLM)** uses the same broad modelling ideas at a smaller scale.

The distinction between:

```text
SLM
and
LLM
```

is largely about scale rather than a completely different architecture.

A smaller model may contain:

- fewer layers
- smaller hidden dimensions
- fewer parameters
- lower compute requirements

This makes SLMs especially useful for:

- local deployment
- experimentation
- embedded systems
- learning model internals
- domain-specific tasks

---

## 1.58 Transformer vs RNN

| RNN | Transformer |
|---|---|
| Recurrent hidden state | Attention-based interaction |
| Sequential processing | Highly parallel training |
| Context carried step by step | Context retrieved directly |
| Long-range paths can be large | Direct token-to-token paths |
| Order inherent in recurrence | Position must be encoded |
| Lower pairwise memory cost | Standard attention is quadratic |

Transformers solve some major RNN limitations while introducing different computational trade-offs.

---

## 1.59 Transformer vs CNN

CNNs exploit:

```text
Local Spatial Structure
```

Transformers use:

```text
Learned Attention Relationships
```

CNN kernels are strongly biased toward local neighbourhoods.

Self-attention can directly connect distant positions.

Modern vision models can also use transformers.

Therefore transformers are not restricted to text.

---

## 1.60 Transformers Beyond Language

Transformer architectures have been applied to:

```text
Images
Audio
Video
Biological Sequences
Time Series
Multimodal Data
```

The general principle is not:

```text
Transformer = Text Model
```

but:

```text
Transformer
= Attention-Based Architecture
for Representing Relationships
Among Structured Elements
```

---

## 1.61 Vision Transformers

A **Vision Transformer (ViT)** divides an image into patches.

Conceptually:

```text
Image
 ↓
Split Into Patches
 ↓
Represent Each Patch as a Vector
 ↓
Treat Patches Like a Sequence
 ↓
Transformer
```

Self-attention then learns relationships among image regions.

This demonstrates how transformer principles extend beyond language.

---

## 1.62 Multimodal Transformers

A multimodal model may process several data types:

```text
Text
Images
Audio
Video
```

Different inputs can be transformed into representations that interact through attention.

Conceptually:

```text
Image Representation
        ↘
         Attention
        ↗
Text Representation
```

This enables models to learn relationships across modalities.

---

## 1.63 Why Transformers Became Dominant

Transformers combine several powerful properties:

```text
Direct Long-Range Relationships

Parallelizable Training

Scalable Matrix Computation

Contextual Representations

Multi-Head Attention

Flexible Architecture
```

These properties align extremely well with modern hardware and large datasets.

Their success is therefore both:

```text
Algorithmic
+
Computational
```

---

## 1.64 Transformers Are Not Magic

Despite their capabilities, transformers still fundamentally perform:

```text
Matrix Multiplication
Softmax
Weighted Sums
Activation Functions
Normalization
Residual Addition
```

repeated across many layers and parameters.

They learn through:

```text
Loss
+
Backpropagation
+
Optimization
```

The scale can be enormous, but the underlying principles connect directly to the foundations already studied.

---

## 1.65 The Complete Conceptual Journey

The progression through neural-network architectures can now be summarized as:

```text
Artificial Neuron
      ↓
Weighted Sum + Activation

Feed-Forward Neural Network
      ↓
Multiple Nonlinear Layers

CNN
      ↓
Exploit Spatial Structure

RNN
      ↓
Introduce Sequential Memory

LSTM / GRU
      ↓
Control Long-Term Memory

Attention
      ↓
Retrieve Relevant Information Directly

Transformer
      ↓
Build Sequence Model Primarily
Around Self-Attention

Large-Scale Transformer
      ↓
Modern Language / Foundation Models
```

Every step solves limitations of earlier structures while building on the same neural-network foundations.

---

## 1.66 Key Takeaways

- Transformers are neural-network architectures built primarily around attention rather than recurrence.
- Their major components include self-attention, feed-forward networks, residual connections, normalization, and positional information.
- Token embeddings provide semantic representations.
- Positional information is necessary because self-attention does not inherently encode order.
- Positional information can be sinusoidal, learned, or relative.
- Transformers use scaled dot-product attention:

```math
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}
{\sqrt{d_k}}
\right)V
```

- Multi-head attention allows multiple relationship spaces to be learned simultaneously.
- Feed-forward networks transform representations independently at each position.
- Attention mixes information across positions; the feed-forward network transforms features within positions.
- Residual connections improve information and gradient flow.
- LayerNorm helps stabilize transformer representations.
- The original Transformer contains encoder and decoder stacks.
- Encoder self-attention can generally use the complete input context.
- Decoder self-attention is causally masked during autoregressive generation.
- Cross-attention allows the decoder to retrieve information from encoder representations.
- Encoder-only transformers are particularly suited to representation and understanding tasks.
- Decoder-only transformers are especially suited to autoregressive generation.
- Encoder-decoder transformers are naturally suited to sequence-to-sequence tasks.
- Decoder-only language models predict tokens autoregressively.
- Next-token prediction commonly uses softmax and cross-entropy.
- Transformers are still ordinary neural networks trained through backpropagation and optimization.
- Transformers parallelize sequence training much better than RNNs.
- Self-attention provides short paths between distant sequence positions.
- Standard self-attention has quadratic complexity in sequence length.
- Transformers have finite context windows rather than unlimited memory.
- Model dimension, number of heads, feed-forward width, and layer count are major transformer hyperparameters.
- AdamW, learning-rate schedules, warm-up, weight decay, and gradient management are commonly important during training.
- Pretraining learns general representations from large datasets.
- Fine-tuning adapts pretrained models to specific tasks.
- Foundation models are broadly reusable pretrained models.
- LLMs are large-scale language models commonly built using transformer architectures.
- SLMs use the same broad principles at smaller computational scale.
- Transformers are used beyond text, including vision, audio, time series, and multimodal systems.
- Transformers are powerful because of architecture, scale, data, and compute—not because they abandon the mathematical foundations of neural networks.

### Memory Hook

```text
TRANSFORMER

=
Attention
+
Feed-Forward Networks
+
Residual Connections
+
LayerNorm
+
Position Information


Attention:
"Who matters to me?"

Feed-Forward:
"What should I do
with what I learned?"


Self-Attention
→ Same Sequence

Cross-Attention
→ One Representation
  Looks at Another


Encoder:
→ Understand / Represent

Decoder:
→ Generate

Encoder + Decoder:
→ Transform One Sequence
  Into Another


RNN:
Carry Context
Step by Step

LSTM:
Carry It Better

Attention:
Retrieve It Directly

Transformer:
Build the Whole Architecture
Around That Retrieval


Language Model:

Previous Tokens
      ↓
Transformer
      ↓
Probability of Next Token


Big Picture:

Neuron
 ↓
Network
 ↓
Deep Network
 ↓
CNN / RNN
 ↓
LSTM / GRU
 ↓
Attention
 ↓
Transformer
 ↓
SLM / LLM


And underneath all of it:

Forward Propagation
        ↓
Loss
        ↓
Backpropagation
        ↓
Optimizer
        ↓
Learning
```