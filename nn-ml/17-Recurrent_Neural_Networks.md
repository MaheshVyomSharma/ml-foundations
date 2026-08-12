# 17. Recurrent Neural Networks

## 1.1 Why Recurrent Neural Networks?

Standard feed-forward neural networks assume that inputs can be processed independently.

For example:

```text
Input
 ↓
Hidden Layers
 ↓
Output
```

But many real-world problems involve **sequences**, where order matters.

Examples include:

- sentences
- speech
- time-series data
- sensor measurements
- stock-price sequences
- biological sequences
- event histories

Consider:

```text
"The dog bit the man"
```

and:

```text
"The man bit the dog"
```

The words are almost identical, but their order completely changes the meaning.

A model for sequential data therefore needs some way to preserve information from earlier elements.

This is the motivation for **Recurrent Neural Networks (RNNs)**.

---

## 1.2 What Is an RNN?

A **Recurrent Neural Network** is a neural-network architecture designed to process sequential data by maintaining information from previous time steps.

Conceptually:

```text
Current Input
     +
Previous State
     ↓
Current State
     ↓
Output
```

Unlike a standard feed-forward network, an RNN contains a recurrent connection.

Information can therefore persist as the sequence is processed.

---

## 1.3 The Central Idea: Memory

Suppose a sequence is:

```text
x₁ → x₂ → x₃ → x₄
```

A feed-forward model could process each value separately.

An RNN instead maintains a **hidden state**:

```text
x₁
 ↓
h₁

x₂ + h₁
 ↓
h₂

x₃ + h₂
 ↓
h₃

x₄ + h₃
 ↓
h₄
```

Each hidden state contains information derived from:

```text
Current Input
+
Previous Hidden State
```

The hidden state therefore acts as the network's memory.

---

## 1.4 Hidden State

At time step:

```math
t
```

the RNN receives:

```math
x_t
```

and the previous hidden state:

```math
h_{t-1}
```

It calculates:

```math
h_t
=
f
\left(
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
\right)
```

where:

- `x_t` = current input
- `h_{t-1}` = previous hidden state
- `h_t` = current hidden state
- `W_xh` = input-to-hidden weights
- `W_hh` = hidden-to-hidden recurrent weights
- `b_h` = hidden bias
- `f` = activation function

The critical new term is:

```math
W_{hh}h_{t-1}
```

because it carries information from the previous step.

---

## 1.5 RNN Output

The output at time:

```math
t
```

can be calculated from the hidden state:

```math
y_t
=
g
\left(
W_{hy}h_t+b_y
\right)
```

where:

- `W_hy` = hidden-to-output weights
- `b_y` = output bias
- `g` = output activation

Thus:

```text
Previous State
      ↓
Current Input → Hidden State → Output
                    ↓
               Next State
```

---

## 1.6 The Recurrent Loop

An RNN is often drawn compactly as:

```text
        ┌───────┐
        │       │
        ↓       │
xₜ → [ RNN ] ──┘
        ↓
       yₜ
```

The loop represents the hidden state being passed forward.

However, this compact diagram can hide what actually happens across time.

To understand RNNs properly, we usually **unroll** the network.

---

## 1.7 Unrolling an RNN

The recurrent network:

```text
        ┌───────┐
        │       │
        ↓       │
xₜ → [ RNN ] ──┘
```

can be expanded across time:

```text
x₁        x₂        x₃        x₄
 ↓         ↓         ↓         ↓
[RNN] → [RNN] → [RNN] → [RNN]
 ↓         ↓         ↓         ↓
y₁        y₂        y₃        y₄
```

The arrows between RNN cells represent hidden states:

```text
h₁ → h₂ → h₃ → h₄
```

This is called an **unrolled RNN**.

---

## 1.8 One Network, Not Many Networks

The unrolled diagram may look like four different neural networks.

It is not.

The same parameters are reused at every time step.

For example:

```math
W_{xh}
```

is the same matrix at:

```text
t = 1
t = 2
t = 3
t = 4
```

Similarly:

```math
W_{hh}
```

is shared across all time steps.

Thus:

```text
Same RNN Cell
+
Same Parameters
+
Repeated Across Time
```

This is analogous to weight sharing in CNNs.

---

## 1.9 Parameter Sharing Across Time

Suppose a sentence contains:

```text
Machine learning is extremely useful
```

The RNN does not use one set of weights for:

```text
Machine
```

and another for:

```text
learning
```

and another for:

```text
useful
```

Instead:

```text
Same Parameters
→ Applied at Every Sequence Position
```

This allows the network to process sequences of varying lengths while keeping parameter count manageable.

---

## 1.10 Initial Hidden State

At the first time step, there is no previous hidden state.

Therefore an initial state is required:

```math
h_0
```

A common choice is:

```math
h_0
=
\mathbf{0}
```

Then:

```math
h_1
=
f
\left(
W_{xh}x_1
+
W_{hh}h_0
+
b_h
\right)
```

Because:

```math
h_0=0
```

the sequence effectively begins without prior information.

In some architectures, the initial state may instead be learned or supplied from another network.

---

## 1.11 Hidden State Accumulates Context

Consider:

```text
"The movie was not good"
```

After reading:

```text
The
```

the hidden state contains little useful meaning.

After:

```text
The movie
```

it contains more context.

After:

```text
The movie was not
```

the hidden state ideally retains the important word:

```text
not
```

When:

```text
good
```

arrives, the model should interpret it using the earlier context.

Conceptually:

```text
Words Arrive Sequentially
        ↓
Hidden State Updated
        ↓
Context Accumulates
        ↓
Later Predictions Use Earlier Information
```

---

## 1.12 Sequence Order Matters

Because the hidden state evolves sequentially:

```math
h_t
=
f(x_t,h_{t-1})
```

changing the order of inputs changes the hidden states.

Therefore:

```text
A → B → C
```

does not generally produce the same representation as:

```text
C → B → A
```

This makes RNNs naturally sensitive to sequence order.

---

## 1.13 RNNs and Time Steps

The index:

```math
t
```

is conventionally called a **time step**.

However, it does not have to represent literal clock time.

For a sentence:

```text
I love neural networks
```

we might have:

```text
t = 1 → I
t = 2 → love
t = 3 → neural
t = 4 → networks
```

Thus:

```text
Time Step
=
Position in Sequence
```

in the general RNN sense.

---

## 1.14 Sequence-to-One Architecture

Some tasks consume a sequence and produce one output.

For example:

```text
"This movie was fantastic"
            ↓
          RNN
            ↓
        Positive
```

Conceptually:

```text
x₁ → x₂ → x₃ → x₄
 ↓    ↓    ↓    ↓
h₁ → h₂ → h₃ → h₄
                 ↓
              Output
```

This is called **many-to-one** or **sequence-to-one**.

Applications include:

- sentiment classification
- sequence classification
- activity recognition

---

## 1.15 One-to-Many Architecture

A single input can also generate a sequence.

Conceptually:

```text
Input
 ↓
RNN
 ↓
y₁ → y₂ → y₃ → y₄
```

This is **one-to-many**.

Historically, examples include generating a sequence from an initial representation.

The central idea is:

```text
One Input
→ Multiple Sequential Outputs
```

---

## 1.16 Many-to-Many Architecture

Both input and output may be sequences.

```text
x₁ → x₂ → x₃ → x₄
 ↓    ↓    ↓    ↓
h₁ → h₂ → h₃ → h₄
 ↓    ↓    ↓    ↓
y₁   y₂   y₃   y₄
```

This is **many-to-many**.

Examples include:

- sequence labelling
- part-of-speech tagging
- some speech tasks

The output at each position can depend on accumulated context.

---

## 1.17 Sequence-to-Sequence

Another many-to-many arrangement may have different input and output lengths.

For example:

```text
Input Sequence
      ↓
Encoder
      ↓
Representation
      ↓
Decoder
      ↓
Output Sequence
```

This architecture became important in machine translation and other sequence-generation tasks.

It is commonly called **sequence-to-sequence**, or **seq2seq**.

Later architectures such as transformers dramatically changed how these tasks are handled, but RNN-based seq2seq models are historically and conceptually important.

---

## 1.18 Input Representation

An RNN requires numerical inputs.

For ordinary numerical time-series data:

```math
x_t
```

may already be a numerical vector.

For words, however, text must first be converted into vectors.

Conceptually:

```text
Word
 ↓
Numerical Representation
 ↓
RNN
```

A common representation is a **word embedding**.

---

## 1.19 Embeddings

Instead of representing words as arbitrary integer IDs, an embedding maps each token to a dense vector.

For example:

```text
"cat"
 ↓
[0.21, -0.47, 0.83, ...]
```

and:

```text
"dog"
 ↓
[0.18, -0.42, 0.79, ...]
```

The vectors can be learned so that related words obtain useful numerical representations.

The RNN then processes the embedding sequence:

```text
Token
 ↓
Embedding
 ↓
RNN
```

---

## 1.20 Hidden State Dimension

The hidden state is a vector:

```math
h_t
\in
\mathbb{R}^{d_h}
```

where:

```math
d_h
```

is the hidden-state dimension.

For example:

```math
d_h=128
```

means each time step maintains a vector containing 128 values.

The hidden-state size is a hyperparameter.

Larger hidden states provide greater representational capacity but increase parameter count and computation.

---

## 1.21 RNN Parameter Dimensions

Suppose:

```math
x_t
\in
\mathbb{R}^{d_x}
```

and:

```math
h_t
\in
\mathbb{R}^{d_h}
```

Then:

```math
W_{xh}
\in
\mathbb{R}^{d_h\times d_x}
```

and:

```math
W_{hh}
\in
\mathbb{R}^{d_h\times d_h}
```

If the output dimension is:

```math
d_y
```

then:

```math
W_{hy}
\in
\mathbb{R}^{d_y\times d_h}
```

The same matrices are reused at every time step.

---

## 1.22 A Simple RNN Cell

A common simple RNN uses:

```math
\tanh
```

for its hidden activation:

```math
h_t
=
\tanh
\left(
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
\right)
```

Why tanh?

Its output lies between:

```math
-1
```

and:

```math
1
```

which keeps hidden-state values bounded.

However, tanh also contributes to the vanishing-gradient problem in long sequences.

---

## 1.23 Forward Propagation Through Time

For a sequence:

```math
x_1,x_2,x_3
```

the forward calculations are:

```math
h_1
=
f
\left(
W_{xh}x_1
+
W_{hh}h_0
+
b_h
\right)
```

then:

```math
h_2
=
f
\left(
W_{xh}x_2
+
W_{hh}h_1
+
b_h
\right)
```

then:

```math
h_3
=
f
\left(
W_{xh}x_3
+
W_{hh}h_2
+
b_h
\right)
```

Thus every state depends recursively on the previous state.

---

## 1.24 Long-Term Dependencies

Suppose the sequence is:

```text
"The book that I bought yesterday after visiting
several stores was excellent."
```

To interpret:

```text
was excellent
```

the network may need to retain information about:

```text
book
```

from many time steps earlier.

This is called a **long-term dependency**.

Basic RNNs often struggle with such dependencies.

---

## 1.25 Why Long-Term Dependencies Are Difficult

Information must repeatedly pass through:

```text
h₁
 ↓
h₂
 ↓
h₃
 ↓
h₄
 ↓
...
 ↓
hₜ
```

At every step, it undergoes another transformation.

As the distance grows, early information can gradually weaken or be overwritten.

During training, the corresponding gradients must also travel backward through all these steps.

This creates a serious optimization problem.

---

## 1.26 Backpropagation Through Time

RNNs are trained using a version of backpropagation called **Backpropagation Through Time (BPTT)**.

First, unroll the RNN:

```text
x₁        x₂        x₃        x₄
 ↓         ↓         ↓         ↓
h₁   →    h₂   →    h₃   →    h₄
 ↓         ↓         ↓         ↓
y₁        y₂        y₃        y₄
```

Then treat the unrolled network as a deep computational graph.

Gradients propagate backward:

```text
h₁ ← h₂ ← h₃ ← h₄
```

Hence the name:

```text
Backpropagation
Through
Time
```

---

## 1.27 Loss Across Time

For many-to-many tasks, a loss may be calculated at every time step.

For example:

```math
J_t
=
L(y_t,\hat{y}_t)
```

The total sequence loss may be:

```math
J
=
\sum_{t=1}^{T}J_t
```

or an average:

```math
J
=
\frac{1}{T}
\sum_{t=1}^{T}J_t
```

BPTT calculates how the shared parameters contributed to losses across the sequence.

---

## 1.28 Shared Parameters Receive Gradients From Many Steps

Because the same recurrent weights are reused:

```math
W_{hh}
```

affects:

```text
h₁
h₂
h₃
...
hₜ
```

Therefore, its total gradient receives contributions from multiple time steps.

Conceptually:

```text
Loss at t₁ ─┐
Loss at t₂ ─┤
Loss at t₃ ─┼→ Gradient for Shared Weights
Loss at t₄ ─┘
```

The optimizer then updates the shared parameter once using the accumulated gradient.

---

## 1.29 RNNs Become Deep Across Time

An RNN may contain only one recurrent layer structurally.

But when unrolled over 100 time steps, the computational graph effectively contains 100 repeated transformations.

Thus:

```text
Sequence Length
→ Effective Computational Depth
```

This explains why gradient problems become particularly severe in recurrent networks.

---

## 1.30 Vanishing Gradients in RNNs

During BPTT, gradients repeatedly pass through recurrent transformations.

A simplified gradient contains repeated products involving:

```math
W_{hh}
```

and activation derivatives.

Conceptually:

```math
\frac{\partial J}{\partial h_1}
```

depends on a chain such as:

```math
\frac{\partial h_2}{\partial h_1}
\frac{\partial h_3}{\partial h_2}
\frac{\partial h_4}{\partial h_3}
\cdots
```

If these factors repeatedly have magnitudes below `1`, the gradient can shrink dramatically.

---

## 1.31 Consequence of Vanishing Gradients

Suppose an important word occurred 50 steps earlier.

If the gradient reaching that time step becomes approximately:

```math
0
```

the network receives almost no learning signal telling it how that earlier information affected the current prediction.

Therefore:

```text
Distant Past
     ↓
Tiny Gradient
     ↓
Weak Learning Signal
     ↓
Long-Term Dependency Not Learned
```

This is one of the fundamental weaknesses of basic RNNs.

---

## 1.32 Exploding Gradients in RNNs

The opposite can also happen.

If repeated recurrent transformations amplify gradients:

```text
Gradient
 ↓
Larger
 ↓
Much Larger
 ↓
Huge
```

the network suffers from **exploding gradients**.

This can produce:

- unstable training
- enormous parameter updates
- divergent loss
- `NaN` values

Gradient clipping is particularly useful in recurrent networks.

---

## 1.33 Gradient Clipping in RNNs

Suppose the gradient is:

```math
g
```

and its norm exceeds:

```math
c
```

We can rescale it:

```math
g_{\text{clipped}}
=
c
\frac{g}{\|g\|}
```

This prevents extremely large gradients from producing catastrophic updates.

Thus:

```text
Exploding Gradient
      ↓
Gradient Clipping
      ↓
Controlled Update
```

Gradient clipping addresses exploding gradients but does not solve the long-term vanishing-gradient problem.

---

## 1.34 Truncated Backpropagation Through Time

For very long sequences, propagating gradients through every previous time step can be computationally expensive.

**Truncated BPTT** limits how far backward gradients are propagated.

Conceptually:

```text
Very Long Sequence

x₁ → x₂ → x₃ → ... → x₁₀₀₀
```

Instead of backpropagating through all 1000 steps at once:

```text
Process Smaller Sequence Windows
→ Backpropagate Through Limited Number of Steps
```

This reduces memory and computational cost.

---

## 1.35 Limitation of Truncated BPTT

Truncation improves practicality but introduces a trade-off.

If gradients are only propagated across short windows, the model receives less direct learning signal for extremely long-range dependencies.

Therefore:

```text
Shorter BPTT Window
→ Lower Computational Cost
→ Less Long-Range Gradient Information
```

It solves a computational problem, not the fundamental memory limitation of a simple RNN.

---

## 1.36 Bidirectional RNNs

Sometimes the entire sequence is available before prediction.

In that case, useful context may exist both before and after the current position.

A **Bidirectional RNN** processes the sequence in both directions.

```text
Forward:
x₁ → x₂ → x₃ → x₄

Backward:
x₁ ← x₂ ← x₃ ← x₄
```

The two representations are then combined.

---

## 1.37 Why Bidirectional Processing Helps

Consider:

```text
"He went to the bank to deposit money."
```

The word:

```text
bank
```

becomes easier to interpret when later words such as:

```text
deposit money
```

are available.

A forward-only RNN at the word `bank` has not yet seen them.

A bidirectional model can use both:

```text
Past Context
+
Future Context
```

when producing its representation.

---

## 1.38 When Bidirectional RNNs Cannot Be Used

Bidirectional processing requires future sequence elements to be available.

Therefore, it is unsuitable for some real-time causal tasks.

For example:

```text
Predict Tomorrow's Sensor Reading
```

cannot use tomorrow's future sensor values as input.

Thus:

```text
Offline Sequence Processing
→ Bidirectional May Be Useful

Strict Real-Time Prediction
→ Future Context Unavailable
```

---

## 1.39 Stacked RNNs

RNNs can also be stacked vertically.

```text
Sequence Input
      ↓
RNN Layer 1
      ↓
RNN Layer 2
      ↓
RNN Layer 3
      ↓
Output
```

At each time step, the hidden representation from one recurrent layer becomes input to the next.

This increases representational capacity.

However, deeper recurrent architectures can be more difficult to train.

---

## 1.40 Dropout in RNNs

Dropout can regularize recurrent networks.

However, applying completely independent dropout masks carelessly across every time step can disrupt temporal information.

Specialized approaches may use consistent or structured masks across sequence positions.

The important principle remains:

```text
Dropout
→ Reduce Overfitting
```

but recurrent structure makes its implementation more delicate than in a simple feed-forward network.

---

## 1.41 Variable-Length Sequences

Real sequences often have different lengths.

For example:

```text
Sentence A
→ 5 tokens

Sentence B
→ 11 tokens

Sentence C
→ 27 tokens
```

To process them efficiently in batches, shorter sequences may be padded.

Example:

```text
A: word word word word <PAD> <PAD>

B: word word word word word word
```

The model must know which positions are real and which are padding.

---

## 1.42 Masking

A **mask** identifies valid sequence positions.

For example:

```text
Sequence:
A B C <PAD> <PAD>

Mask:
1 1 1 0 0
```

The padded positions can then be ignored when calculating relevant outputs or losses.

This allows variable-length sequences to coexist within a batch.

---

## 1.43 RNNs for Time-Series Data

RNNs are not limited to language.

Suppose we have:

```text
Temperature:
t₁ → t₂ → t₃ → t₄ → ...
```

or:

```text
Machine Sensor:
x₁ → x₂ → x₃ → x₄ → ...
```

The RNN can use previous observations to build a hidden representation of recent history.

Applications can include:

- forecasting
- anomaly detection
- sequence classification
- sensor monitoring

---

## 1.44 RNNs vs Ordinary Dense Networks

| Dense Network | RNN |
|---|---|
| No built-in sequential memory | Maintains hidden state |
| Inputs often treated independently | Order matters |
| No recurrent connection | Recurrent hidden connection |
| Fixed feed-forward computation | Repeated computation across time |
| Parameters used once per layer | Parameters shared across time |

The RNN adds explicit sequential state to the neural-network framework.

---

## 1.45 RNNs vs CNNs

CNNs and RNNs exploit different structures.

```text
CNN
→ Spatial Locality

RNN
→ Sequential Dependency
```

CNN:

```text
Nearby Pixels
→ Local Spatial Patterns
```

RNN:

```text
Previous Sequence Elements
→ Context for Current Element
```

Both use parameter sharing:

```text
CNN
→ Share Kernel Across Space

RNN
→ Share Recurrent Weights Across Time
```

This is a useful connection between the two architectures.

---

## 1.46 The Fundamental Weakness of Basic RNNs

The simple RNN theoretically has access to all previous hidden states through recurrence.

But practically:

```text
Recent Information
→ Easier to Preserve

Very Old Information
→ Difficult to Preserve
```

because:

- hidden states are repeatedly transformed
- information can be overwritten
- gradients can vanish across long sequences

This is the key problem that motivates **LSTM** and **GRU** architectures.

---

## 1.47 Why Not Simply Make the Hidden State Larger?

Increasing:

```math
d_h
```

gives the network more representational capacity.

But it does not remove the fundamental gradient problem.

A larger simple RNN can still suffer from:

```text
Vanishing Gradients
Exploding Gradients
Long-Term Dependency Failure
```

The solution requires changing how information flows through the recurrent cell itself.

---

## 1.48 From RNN to LSTM

The basic RNN repeatedly transforms:

```math
h_{t-1}
```

into:

```math
h_t
```

through the same nonlinear recurrence.

LSTM introduces a more carefully controlled memory pathway.

Instead of forcing all information through one hidden-state transformation, it introduces:

```text
Cell State
+
Gates
```

The gates learn:

```text
What to Forget
What to Store
What to Output
```

This gives the network better control over long-term information.

---

## 1.49 From RNN to GRU

GRU follows a similar motivation but uses a somewhat simpler architecture.

Instead of the full LSTM gate structure, it uses mechanisms such as:

```text
Update Gate
Reset Gate
```

Both LSTM and GRU are designed to improve recurrent learning over longer dependencies.

They deserve separate treatment because their internal mechanics are central to understanding recurrent deep learning.

---

## 1.50 RNNs and Modern Sequence Models

RNNs, LSTMs, and GRUs were dominant sequence architectures for many years.

Modern large language models predominantly use **transformers** rather than recurrent networks.

Transformers address sequence relationships using attention mechanisms rather than passing one recurrent hidden state sequentially through every position.

However, understanding RNNs remains valuable because they make several concepts explicit:

- sequence state
- temporal dependency
- parameter sharing across time
- long-term dependency problems
- sequence-to-sequence modelling
- why newer architectures were needed

RNN-family models also remain useful for some sequence and time-series applications.

---

## 1.51 RNN Training Loop

The overall training process remains familiar:

```text
Sequence Batch
      ↓
Forward Through Time
      ↓
Predictions
      ↓
Sequence Loss
      ↓
Backpropagation Through Time
      ↓
Gradients
      ↓
Optional Gradient Clipping
      ↓
Optimizer
      ↓
Update Shared Parameters
```

The fundamental neural-network learning machinery has not changed.

The computational graph has simply acquired a sequential dimension.

---

## 1.52 The Big Picture

The simplest RNN can be summarized as:

```text
             Previous Memory
                   ↓
Current Input → RNN Cell
                   ↓
             Current Memory
                   ↓
                Output
```

Repeated through time:

```text
x₁       x₂       x₃       x₄
 ↓        ↓        ↓        ↓
h₁  →    h₂  →    h₃  →    h₄
 ↓        ↓        ↓        ↓
y₁       y₂       y₃       y₄
```

The central mechanism is:

```math
h_t
=
f
\left(
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
\right)
```

Everything distinctive about the basic RNN follows from the presence of:

```math
h_{t-1}
```

in the calculation of the current state.

---

## 1.53 Key Takeaways

- RNNs are neural networks designed for sequential data.
- Sequence order matters in RNNs.
- RNNs maintain a hidden state that acts as memory.
- The current hidden state depends on the current input and previous hidden state:

```math
h_t
=
f
\left(
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
\right)
```

- The hidden state carries context forward through the sequence.
- RNNs are commonly visualized by unrolling them through time.
- The unrolled cells share the same parameters.
- Parameter sharing keeps model size independent of sequence length.
- `h_0` is commonly initialized to zero.
- Time steps represent sequence positions and need not correspond to literal time.
- RNN architectures include many-to-one, one-to-many, and many-to-many arrangements.
- Sequence-to-sequence architectures map one sequence to another.
- Text inputs are commonly converted to numerical embeddings before entering an RNN.
- Hidden-state size is a hyperparameter.
- Simple RNNs commonly use tanh hidden activations.
- RNN forward propagation recursively computes hidden states through time.
- Long-term dependencies require information from distant earlier sequence positions.
- Basic RNNs struggle with long-term dependencies.
- RNNs are trained using Backpropagation Through Time.
- Shared recurrent parameters receive gradient contributions from multiple time steps.
- Long sequences make RNN computational graphs effectively very deep.
- Vanishing gradients prevent strong learning from distant time steps.
- Exploding gradients can make RNN training unstable.
- Gradient clipping can control exploding gradients.
- Truncated BPTT limits computational cost for long sequences.
- Bidirectional RNNs use both past and future context when the full sequence is available.
- Stacked RNNs add recurrent depth.
- Padding and masking allow variable-length sequences to be batched.
- CNNs share weights across space; RNNs share weights across time.
- Increasing hidden-state size does not solve the fundamental long-term gradient problem.
- LSTM and GRU modify the recurrent cell to preserve information more effectively.
- Transformers now dominate many large-scale sequence tasks, but RNNs remain foundational for understanding sequence modelling.

### Memory Hook

```text
RNN
= Neural Network With Memory

At Time t:

Current Input
+
Previous Hidden State
        ↓
Current Hidden State
        ↓
Output


Core Equation:

hₜ
=
f(
input contribution
+
previous-state contribution
+
bias
)


Hidden State
→ Memory / Context

Same Weights
→ Reused Across Time


Architectures:

Many-to-One
→ Sequence → One Output

One-to-Many
→ One Input → Sequence

Many-to-Many
→ Sequence → Sequence


Training:

Forward Propagation
→ Through Time

Backpropagation
→ Through Time
→ BPTT


Big Problem:

Long Sequence
      ↓
Repeated Multiplication
      ↓
Vanishing / Exploding Gradients
      ↓
Basic RNN Struggles With
Long-Term Dependencies


CNN
→ Shares Weights Across Space

RNN
→ Shares Weights Across Time


What Comes Next?

RNN:
"I have memory."

LSTM / GRU:
"I need better control
over what I remember
and what I forget."
```