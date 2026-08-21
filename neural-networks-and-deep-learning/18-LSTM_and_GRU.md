# 18. LSTM and GRU

## 1. Why Basic RNNs Need Improvement

A basic RNN updates its hidden state using:

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

This gives the network memory, but the hidden state is repeatedly transformed at every time step.

For a long sequence:

```text
h₁ → h₂ → h₃ → h₄ → ... → h₁₀₀
```

information from early time steps must survive many transformations.

During backpropagation, gradients must travel through the same long chain in reverse.

This can cause:

```text
Vanishing Gradients
        ↓
Weak Learning From Distant Past

or

Exploding Gradients
        ↓
Unstable Training
```

Basic RNNs therefore struggle with **long-term dependencies**.

---

## 2. The Core Idea Behind LSTM and GRU

LSTM and GRU improve recurrent networks by controlling the flow of information.

Instead of simply transforming the previous hidden state at every step, they introduce **gates**.

A gate learns whether information should be:

```text
Kept
Forgotten
Updated
Exposed as Output
```

Conceptually:

```text
Previous Information
        ↓
      Gates
   ↙    ↓    ↘
Forget Keep Update
        ↓
New State
```

These gates are themselves learned neural-network components.

---

## 3. What Is a Gate?

A gate is usually based on the sigmoid function:

```math
\sigma(z)
=
\frac{1}{1+e^{-z}}
```

which produces values between:

```math
0
```

and:

```math
1
```

This makes sigmoid useful as a soft control mechanism.

Conceptually:

```text
Gate ≈ 0
→ Block Information

Gate ≈ 1
→ Allow Information

Gate Between 0 and 1
→ Allow Partially
```

The gate is differentiable, so its behaviour can be learned through backpropagation.

---

## 4. Long Short-Term Memory

**LSTM** stands for:

```text
Long Short-Term Memory
```

An LSTM maintains two important states:

```text
Hidden State
hₜ

Cell State
Cₜ
```

The hidden state represents the information exposed to the rest of the network.

The cell state acts as a more persistent internal memory pathway.

Conceptually:

```text
Previous Cell State Cₜ₋₁
          ↓
     Controlled Update
          ↓
Current Cell State Cₜ
          ↓
     Hidden State hₜ
```

The cell state is the central feature that distinguishes LSTM from a simple RNN.

---

## 5. The LSTM Cell

A standard LSTM contains three major gates:

```text
Forget Gate
Input Gate
Output Gate
```

It also computes a candidate memory value.

The overall flow is:

```text
Previous Cell State
        ↓
   Forget Gate
        ↓
Retained Memory
        +
New Candidate Memory
        ↑
    Input Gate
        ↓
New Cell State
        ↓
   Output Gate
        ↓
New Hidden State
```

Each gate controls a different aspect of memory.

---

## 6. Inputs to an LSTM Cell

At time step:

```math
t
```

the LSTM receives:

```math
x_t
```

and previous hidden state:

```math
h_{t-1}
```

These are often combined conceptually as:

```math
[h_{t-1},x_t]
```

The gates examine both:

```text
Current Input
+
Previous Hidden Context
```

before deciding what information should flow through the cell.

---

## 7. Forget Gate

The **forget gate** decides how much of the previous cell state should be retained.

It is commonly calculated as:

```math
f_t
=
\sigma
\left(
W_f[h_{t-1},x_t]
+
b_f
\right)
```

where:

```math
f_t
```

contains values between `0` and `1`.

The previous memory:

```math
C_{t-1}
```

is multiplied element-wise by:

```math
f_t
```

Conceptually:

```text
Old Memory
   ×
Forget Gate
   ↓
Retained Old Memory
```

---

## 8. Forget Gate Interpretation

Suppose one component of the forget gate produces:

```math
f_t=0.95
```

Then most of that memory component is retained.

If:

```math
f_t=0.05
```

most of it is removed.

Thus:

```text
fₜ ≈ 1
→ Remember

fₜ ≈ 0
→ Forget
```

The network learns these decisions from data.

---

## 9. Input Gate

The **input gate** decides how much new information should be written into the cell state.

It is commonly calculated as:

```math
i_t
=
\sigma
\left(
W_i[h_{t-1},x_t]
+
b_i
\right)
```

The gate controls whether newly computed candidate information should be stored.

Conceptually:

```text
New Information
      ↓
 Input Gate
      ↓
How Much Should Be Stored?
```

---

## 10. Candidate Cell State

The LSTM also generates candidate memory:

```math
\tilde{C}_t
=
\tanh
\left(
W_C[h_{t-1},x_t]
+
b_C
\right)
```

This represents possible new information that could be added to the memory.

The input gate then determines how much of this candidate is actually stored:

```math
i_t
\odot
\tilde{C}_t
```

where:

```math
\odot
```

denotes element-wise multiplication.

---

## 11. Updating the Cell State

The new cell state combines:

```text
Retained Old Memory
+
Selected New Memory
```

Mathematically:

```math
C_t
=
f_t
\odot
C_{t-1}
+
i_t
\odot
\tilde{C}_t
```

This is the heart of the LSTM.

The first term says:

```text
What Should I Keep From the Past?
```

The second says:

```text
What New Information Should I Add?
```

---

## 12. Why the Cell-State Update Helps

Notice that the previous cell state enters through an additive pathway:

```math
C_t
=
f_t
\odot
C_{t-1}
+
\text{new information}
```

The memory is not forced through a completely new nonlinear transformation at every step.

This creates a more direct path through time.

Conceptually:

```text
C₁ ─────→ C₂ ─────→ C₃ ─────→ C₄
  controlled   controlled   controlled
   updates      updates      updates
```

This helps information and gradients survive across longer sequences.

---

## 13. Output Gate

The **output gate** determines how much of the current cell state should be exposed as the hidden state.

It is calculated as:

```math
o_t
=
\sigma
\left(
W_o[h_{t-1},x_t]
+
b_o
\right)
```

The hidden state becomes:

```math
h_t
=
o_t
\odot
\tanh(C_t)
```

Thus:

```text
Cell State
   ↓
tanh
   ↓
Output Gate
   ↓
Hidden State
```

The internal memory and exposed output are therefore related but distinct.

---

## 14. Cell State vs Hidden State

This distinction is central.

### 14.1. Cell State

```math
C_t
```

acts as long-term internal memory.

### 14.2. Hidden State

```math
h_t
```

is the current exposed representation passed:

- to the next time step
- to higher layers
- potentially to the output layer

A useful intuition is:

```text
Cell State
→ Internal Memory

Hidden State
→ Current Working Representation
```

---

## 15. The Complete LSTM Equations

A standard LSTM can be summarized as:

### 15.1. Forget Gate

```math
f_t
=
\sigma
\left(
W_f[h_{t-1},x_t]
+
b_f
\right)
```

### 15.2. Input Gate

```math
i_t
=
\sigma
\left(
W_i[h_{t-1},x_t]
+
b_i
\right)
```

### 15.3. Candidate Memory

```math
\tilde{C}_t
=
\tanh
\left(
W_C[h_{t-1},x_t]
+
b_C
\right)
```

### 15.4. Cell-State Update

```math
C_t
=
f_t
\odot
C_{t-1}
+
i_t
\odot
\tilde{C}_t
```

### 15.5. Output Gate

```math
o_t
=
\sigma
\left(
W_o[h_{t-1},x_t]
+
b_o
\right)
```

### 15.6. Hidden State

```math
h_t
=
o_t
\odot
\tanh(C_t)
```

These equations describe the standard LSTM memory mechanism.

---

## 16. LSTM Memory Intuition

Consider the sentence:

```text
"I grew up in France ... I speak fluent French."
```

The word:

```text
France
```

may occur many time steps before:

```text
French
```

An LSTM can learn:

```text
France
 ↓
Store Relevant Information
 ↓
Carry It Through Many Steps
 ↓
Use It When Predicting Later Words
```

The forget and input gates determine whether this information remains relevant as the sequence progresses.

---

## 17. Forgetting Is Useful

A good memory system must not remember everything forever.

Suppose the sequence changes topic.

Information that was previously important may become irrelevant.

The forget gate allows:

```text
Old Relevant Context
→ Retain

Old Irrelevant Context
→ Remove
```

This prevents the cell state from becoming an uncontrolled accumulation of all previous information.

---

## 18. LSTM and Gradient Flow

The important cell-state pathway:

```math
C_{t-1}
\rightarrow
C_t
```

allows gradients to propagate through a more controlled route.

The derivative includes the forget gate:

```math
\frac{\partial C_t}
{\partial C_{t-1}}
=
f_t
```

If:

```math
f_t
\approx1
```

the gradient can pass backward with relatively little attenuation.

Thus the network can learn to preserve gradient flow when long-term memory is useful.

---

## 19. Does LSTM Eliminate Vanishing Gradients?

No.

LSTM substantially reduces the long-term dependency problem, but it does not mathematically guarantee that gradients can never vanish or explode.

Training can still be affected by:

- very long sequences
- poor optimization
- poor initialization
- inappropriate learning rates
- unstable gradients

Gradient clipping may still be used.

The important point is:

```text
LSTM
→ Makes Long-Term Gradient Flow Much Easier

not

LSTM
→ Makes Gradient Problems Impossible
```

---

## 20. LSTM Parameter Cost

A basic RNN computes roughly one main recurrent transformation.

An LSTM computes several:

```text
Forget Gate
Input Gate
Candidate Memory
Output Gate
```

Therefore, an LSTM contains substantially more parameters than a basic RNN with the same hidden dimension.

Conceptually:

```text
Basic RNN
→ Simpler
→ Fewer Parameters

LSTM
→ More Complex
→ More Parameters
→ Better Long-Term Memory Control
```

This is the price of its improved memory mechanism.

---

## 21. Gated Recurrent Unit

The **Gated Recurrent Unit**, or **GRU**, is another gated recurrent architecture.

It was designed with similar goals to LSTM:

```text
Preserve Important Information
Forget Irrelevant Information
Improve Long-Term Dependencies
Improve Gradient Flow
```

However, GRU uses a simpler structure.

It generally combines some of the roles handled separately in LSTM.

---

## 22. Main Difference Between LSTM and GRU

LSTM maintains:

```text
Cell State Cₜ
+
Hidden State hₜ
```

GRU typically maintains only:

```text
Hidden State hₜ
```

The hidden state itself serves as the primary memory.

GRU also uses fewer gates.

Typical GRU gates are:

```text
Update Gate
Reset Gate
```

Thus:

```text
LSTM
→ More Explicit Memory Structure

GRU
→ More Compact Gated Structure
```

---

## 23. Update Gate

The **update gate** determines how much of the previous hidden state should be retained.

It is commonly calculated as:

```math
z_t
=
\sigma
\left(
W_zx_t
+
U_zh_{t-1}
+
b_z
\right)
```

The update gate performs a role somewhat analogous to combining the LSTM's:

```text
Forget Decision
+
Input Decision
```

into one mechanism.

---

## 24. Update Gate Intuition

If:

```math
z_t
\approx1
```

the GRU can retain a large amount of the previous hidden state.

If:

```math
z_t
\approx0
```

the GRU can replace more of the previous state with newly computed information.

Thus:

```text
Large Update Gate
→ Preserve Old State

Small Update Gate
→ Favor New State
```

Exact conventions may differ slightly across formulations, but the conceptual role remains the same.

---

## 25. Reset Gate

The **reset gate** determines how much previous information should influence the candidate new state.

It is commonly calculated as:

```math
r_t
=
\sigma
\left(
W_rx_t
+
U_rh_{t-1}
+
b_r
\right)
```

The reset gate controls how strongly:

```math
h_{t-1}
```

contributes when generating new candidate information.

---

## 26. Reset Gate Intuition

Suppose the sequence moves into a context where earlier information is no longer useful.

The reset gate can reduce the influence of the previous hidden state.

Conceptually:

```text
Previous State
     ↓
 Reset Gate
     ↓
How Much Old Context
Should Influence New Candidate?
```

Thus:

```text
Reset Strongly
→ Ignore Much of the Past

Reset Weakly
→ Use More Past Context
```

---

## 27. Candidate Hidden State

The GRU computes a candidate hidden state:

```math
\tilde{h}_t
=
\tanh
\left(
W_hx_t
+
U_h
\left(
r_t\odot h_{t-1}
\right)
+
b_h
\right)
```

The reset gate controls how much of the previous hidden state contributes to this candidate.

The candidate represents possible new information for the current state.

---

## 28. Updating the GRU Hidden State

The final hidden state combines:

```text
Previous Hidden State
+
Candidate Hidden State
```

using the update gate.

A common formulation is:

```math
h_t
=
z_t\odot h_{t-1}
+
(1-z_t)\odot\tilde{h}_t
```

This creates a direct pathway from:

```math
h_{t-1}
```

to:

```math
h_t
```

which helps preserve information across time.

---

## 29. The Complete GRU Equations

A common GRU formulation is:

### 29.1. Update Gate

```math
z_t
=
\sigma
\left(
W_zx_t
+
U_zh_{t-1}
+
b_z
\right)
```

### 29.2. Reset Gate

```math
r_t
=
\sigma
\left(
W_rx_t
+
U_rh_{t-1}
+
b_r
\right)
```

### 29.3. Candidate State

```math
\tilde{h}_t
=
\tanh
\left(
W_hx_t
+
U_h
\left(
r_t\odot h_{t-1}
\right)
+
b_h
\right)
```

### 29.4. Hidden-State Update

```math
h_t
=
z_t\odot h_{t-1}
+
(1-z_t)\odot\tilde{h}_t
```

Different libraries may use slightly different notation or gate conventions, but the underlying principle is the same.

---

## 30. GRU Memory Intuition

The update mechanism can be viewed as:

```text
Old Memory
   ↓
Update Gate
   ↓
Keep Some Old Information
        +
Add Some New Candidate
        ↓
New Hidden State
```

Instead of keeping a separate long-term cell state, GRU integrates memory directly into the hidden-state update.

---

## 31. LSTM vs GRU Structure

A simplified comparison:

| LSTM | GRU |
|---|---|
| Cell state + hidden state | Hidden state only |
| Forget gate | No separate forget gate |
| Input gate | Combined largely into update mechanism |
| Output gate | No separate output gate |
| More parameters | Fewer parameters |
| More complex | Simpler |

Both are gated recurrent networks.

---

## 32. LSTM vs GRU Performance

There is no universal winner.

For some tasks:

```text
LSTM
→ performs better
```

For others:

```text
GRU
→ performs equally well or better
```

Performance depends on:

- dataset
- sequence length
- hidden-state size
- training setup
- task complexity

Thus:

```text
LSTM > GRU
```

is not a universal rule.

Neither is:

```text
GRU > LSTM
```

The choice should be validated experimentally.

---

## 33. Why GRU Can Train Faster

GRU has fewer gates and usually fewer parameters than an equivalent LSTM.

This can mean:

```text
Fewer Matrix Operations
        ↓
Lower Computational Cost
        ↓
Potentially Faster Training
```

GRU can therefore be attractive when:

- computational resources are limited
- the dataset is moderate in size
- LSTM complexity is unnecessary

---

## 34. Why LSTM Can Be More Flexible

LSTM separately controls:

```text
Forget
Write
Output
```

and maintains a distinct cell state.

This gives the architecture fine-grained control over information flow.

For some tasks involving complex long-term dependencies, this additional flexibility may be useful.

However, greater architectural complexity does not automatically guarantee better performance.

---

## 35. Basic RNN vs LSTM vs GRU

| Feature | Basic RNN | LSTM | GRU |
|---|---|---|---|
| Hidden state | Yes | Yes | Yes |
| Separate cell state | No | Yes | No |
| Gates | No | 3 major gates | 2 major gates |
| Long-term dependency handling | Weak | Stronger | Stronger |
| Parameter count | Lowest | Highest | Intermediate |
| Training complexity | Lowest | Highest | Intermediate |
| Vanishing-gradient resistance | Weak | Improved | Improved |

This is the central comparison to remember.

---

## 36. Why Gates Improve Long-Term Memory

A basic RNN repeatedly performs something like:

```math
h_t
=
\tanh
\left(
W_xx_t
+
W_hh_{t-1}
+
b
\right)
```

The entire memory is transformed again at every step.

Gated architectures instead allow direct weighted retention:

```text
Old State
×
Gate
+
New Information
```

This gives the model explicit control over how much old information survives.

That direct retention mechanism is the major conceptual breakthrough.

---

## 37. Gates Are Learned, Not Manually Programmed

The network is not told rules such as:

```text
Remember nouns
Forget adjectives
Keep information for 20 steps
```

Instead, gate parameters are learned through:

```text
Forward Propagation
        ↓
Loss
        ↓
Backpropagation Through Time
        ↓
Gradient Updates
```

The model learns what information is useful for minimizing its loss.

---

## 38. Gates Are Vectors

A gate is not usually one single scalar value.

For a hidden dimension:

```math
d_h
```

the gate is typically a vector:

```math
f_t
\in
\mathbb{R}^{d_h}
```

This means different memory components can be controlled independently.

For example:

```text
Memory Component 1
→ Keep 95%

Memory Component 2
→ Keep 10%

Memory Component 3
→ Keep 60%
```

The network therefore performs fine-grained memory management.

---

## 39. Element-Wise Multiplication

Gate application uses element-wise multiplication:

```math
f_t
\odot
C_{t-1}
```

This means corresponding elements are multiplied:

```math
\begin{bmatrix}
f_1 \\
f_2 \\
f_3
\end{bmatrix}
\odot
\begin{bmatrix}
C_1 \\
C_2 \\
C_3
\end{bmatrix}
=
\begin{bmatrix}
f_1C_1 \\
f_2C_2 \\
f_3C_3
\end{bmatrix}
```

Thus each memory dimension can be retained or suppressed separately.

---

## 40. Why Sigmoid Is Used for Gates

Sigmoid produces:

```math
0<\sigma(z)<1
```

which naturally behaves like a continuous switch.

Compare:

```text
0
→ Completely Block

0.25
→ Mostly Block

0.75
→ Mostly Allow

1
→ Completely Allow
```

A step function could provide hard decisions but would be difficult to train with gradient-based methods.

Sigmoid provides a smooth, differentiable gate.

---

## 41. Why Tanh Is Used for Candidate Values

Candidate memory values often use:

```math
\tanh
```

which produces values in:

```math
(-1,1)
```

This allows candidate information to contain:

```text
Positive Contributions
or
Negative Contributions
```

while keeping its magnitude bounded.

Thus a common gated-cell pattern is:

```text
Sigmoid
→ How Much?

Tanh
→ What Information?
```

---

## 42. LSTM Through Time

Across multiple time steps:

```text
x₁       x₂       x₃       x₄
 ↓        ↓        ↓        ↓
LSTM →   LSTM →   LSTM →   LSTM
 ↓        ↓        ↓        ↓
h₁       h₂       h₃       h₄

C₁ ───→ C₂ ───→ C₃ ───→ C₄
```

Both:

```math
h_t
```

and:

```math
C_t
```

move forward through the sequence.

The same LSTM parameters are reused at every time step.

---

## 43. GRU Through Time

GRU has a simpler state flow:

```text
x₁      x₂      x₃      x₄
 ↓       ↓       ↓       ↓
GRU →   GRU →   GRU →   GRU
 ↓       ↓       ↓       ↓
h₁ ───→ h₂ ───→ h₃ ───→ h₄
```

There is no separate cell-state pathway.

The hidden state itself carries the recurrent memory.

---

## 44. Bidirectional LSTM

LSTMs can also be bidirectional.

One LSTM processes:

```text
Left → Right
```

while another processes:

```text
Right → Left
```

Their representations are combined.

Conceptually:

```text
Past Context
      +
Future Context
      ↓
Richer Representation
```

This is useful when the entire sequence is available.

---

## 45. Bidirectional GRU

GRUs can similarly be bidirectional.

The same restriction applies:

```text
Future Context Must Be Available
```

Therefore, bidirectional recurrent models are useful for offline sequence processing but unsuitable when predictions must depend only on information available up to the current time.

---

## 46. Stacked LSTM and GRU Networks

LSTM and GRU layers can be stacked:

```text
Sequence Input
      ↓
LSTM Layer 1
      ↓
LSTM Layer 2
      ↓
LSTM Layer 3
      ↓
Output
```

or:

```text
Sequence Input
      ↓
GRU Layer 1
      ↓
GRU Layer 2
      ↓
Output
```

Lower layers can learn relatively local sequence patterns.

Higher layers can learn more abstract sequential representations.

---

## 47. Sequence Output Choices

An LSTM or GRU may return:

```text
Final Hidden State Only
```

or:

```text
Hidden State at Every Time Step
```

The correct choice depends on the task.

### 47.1. Sequence-to-One

```text
Whole Sequence
→ Final State
→ Classification
```

### 47.2. Many-to-Many

```text
Every Time Step
→ Hidden State
→ Output
```

This mirrors the architecture patterns already seen with basic RNNs.

---

## 48. LSTM for Sequence Classification

Suppose the task is sentiment classification:

```text
"This movie was unexpectedly brilliant"
```

The sequence is processed:

```text
Word Embeddings
      ↓
LSTM
      ↓
Final Hidden Representation
      ↓
Dense Layer
      ↓
Sigmoid
      ↓
Positive Probability
```

The LSTM attempts to compress useful sequence context into the final representation.

---

## 49. LSTM for Time-Series Data

For a sensor sequence:

```text
x₁ → x₂ → x₃ → ... → xₜ
```

the LSTM can retain relevant historical patterns.

The final state may be used for:

```text
Forecasting
Classification
Anomaly Detection
```

The architecture is not inherently linguistic.

LSTM and GRU are general sequence models.

---

## 50. Backpropagation Through Time Still Applies

LSTM and GRU are still recurrent neural networks.

They are trained using BPTT.

Conceptually:

```text
Forward Through Sequence
        ↓
Compute Loss
        ↓
Backpropagate Through Time
        ↓
Compute Gate and Weight Gradients
        ↓
Optimizer
        ↓
Update Parameters
```

The gating mechanism improves information flow, but the fundamental learning algorithm remains gradient-based.

---

## 51. Gradient Clipping Can Still Be Useful

Even gated recurrent networks can experience large gradients.

Therefore:

```text
LSTM / GRU
+
Gradient Clipping
```

is a common combination.

The gates primarily help with preserving useful information and reducing vanishing-gradient problems.

Gradient clipping directly limits exploding gradients.

---

## 52. Dropout With LSTM and GRU

Dropout can regularize LSTM and GRU networks.

It may be applied to:

- inputs
- outputs between recurrent layers
- recurrent connections using specialized approaches

Because temporal continuity matters, dropout must be applied with care.

Frameworks often provide recurrent dropout options specifically designed for recurrent architectures.

---

## 53. LSTM and GRU Hyperparameters

Important hyperparameters include:

```text
Hidden-State Size
Number of Recurrent Layers
Bidirectional or Unidirectional
Dropout Rate
Learning Rate
Batch Size
Sequence Length
Optimizer
```

The choice between:

```text
LSTM
or
GRU
```

is itself an architectural hyperparameter.

---

## 54. Computational Trade-Off

The architecture choice can be summarized as:

```text
Basic RNN
→ Cheapest
→ Weakest Long-Term Memory

GRU
→ Moderate Complexity
→ Stronger Memory

LSTM
→ Greater Complexity
→ Fine-Grained Memory Control
```

This is a useful intuition rather than a strict ranking of prediction quality.

---

## 55. LSTM and GRU vs Transformers

LSTM and GRU process sequence elements recurrently:

```text
x₁ → x₂ → x₃ → x₄
```

Each step depends on earlier state.

Transformers instead use **attention** to directly model relationships between sequence positions.

Conceptually:

```text
RNN Family
→ Carry Information Step by Step

Transformer
→ Allow Positions to Attend Directly to Other Positions
```

This difference has major computational consequences.

---

## 56. Sequential Computation Limitation

Because:

```math
h_t
```

depends on:

```math
h_{t-1}
```

the recurrent computation is inherently sequential.

To compute:

```math
h_{100}
```

we must first compute:

```text
h₁
→ h₂
→ h₃
→ ...
→ h₉₉
```

This limits parallelization during sequence processing.

Transformers avoid much of this recurrent dependency and can process sequence positions more parallelly during training.

---

## 57. Why LSTM and GRU Still Matter

Despite transformers dominating many modern language tasks, LSTM and GRU remain valuable.

They are useful for understanding:

- memory in neural networks
- sequence state
- gating
- long-term dependency problems
- recurrent computation
- gradient flow through time

They can also remain practical for:

- smaller sequence problems
- time-series modelling
- streaming data
- resource-constrained systems
- tasks where recurrent state is naturally useful

---

## 58. The Evolution of Sequence Models

A useful historical and conceptual progression is:

```text
Basic RNN
    ↓
Has Memory
but
Long-Term Gradient Problems
    ↓
LSTM
    ↓
Explicit Cell State + Gates
    ↓
GRU
    ↓
Simplified Gated Memory
    ↓
Attention
    ↓
Transformers
```

Each step attempts to improve how models represent dependencies across sequences.

---

## 59. The Most Important LSTM Intuition

Do not reduce LSTM to memorizing six equations.

The key mechanism is:

```text
Old Memory
      ↓
Forget What Is Irrelevant
      ↓
Add Useful New Information
      ↓
Maintain Updated Memory
      ↓
Expose What Is Needed
```

That is the conceptual model from which the equations follow.

---

## 60. The Most Important GRU Intuition

GRU asks essentially:

```text
How Much of the Old State
Should I Keep?

How Much New Information
Should Replace It?
```

and uses fewer mechanisms to answer those questions.

Thus:

```text
LSTM
→ Detailed Memory Management

GRU
→ Simplified Memory Management
```

---

## 61. Key Takeaways

- Basic RNNs struggle with long-term dependencies because information and gradients must pass through many repeated transformations.
- LSTM and GRU are gated recurrent architectures designed to improve long-term information flow.
- Gates use sigmoid activations to control information continuously between `0` and `1`.
- LSTM maintains both a hidden state and a separate cell state.
- The cell state provides a relatively direct memory pathway across time.
- LSTM contains forget, input, and output gates.
- The forget gate controls how much old memory is retained.
- The input gate controls how much new candidate memory is stored.
- The output gate controls how much cell-state information is exposed through the hidden state.
- The LSTM cell-state update is:

```math
C_t
=
f_t\odot C_{t-1}
+
i_t\odot\tilde{C}_t
```

- LSTM improves gradient flow but does not completely eliminate gradient problems.
- GRU provides a simpler gated recurrent architecture.
- GRU generally uses an update gate and reset gate.
- GRU normally maintains only a hidden state rather than a separate cell state.
- The update gate controls the balance between previous state and new candidate information.
- The reset gate controls how much previous state influences the candidate.
- GRU usually contains fewer parameters than LSTM.
- LSTM offers more explicit and fine-grained memory control.
- Neither LSTM nor GRU is universally superior.
- Both can be bidirectional or stacked.
- Both are trained using Backpropagation Through Time.
- Gradient clipping can still be useful.
- LSTM and GRU remain sequential architectures and therefore have limited parallelism compared with transformers.
- Transformers replaced recurrent architectures in many large-scale sequence applications but do not make the recurrent concepts obsolete.
- The central purpose of gated recurrence is to learn what information should be remembered, forgotten, and updated.

### 61.1. Memory Hook

```text
Basic RNN:
"I remember,
but old information fades."


LSTM:
"I control memory explicitly."

Forget Gate
→ What should I erase?

Input Gate
→ What should I write?

Cell State
→ What should I remember?

Output Gate
→ What should I expose?


LSTM State:

Cell State
→ Long-Term Memory

Hidden State
→ Current Working State


GRU:
"Do the same job
with fewer moving parts."

Update Gate
→ Old vs New

Reset Gate
→ How Much Past
  Should Affect New Candidate?


Comparison:

RNN
→ No Gates

GRU
→ 2 Main Gates

LSTM
→ 3 Main Gates
+ Separate Cell State


Core Idea:

Gates are learned valves
that control
the flow of information
through time.
```
