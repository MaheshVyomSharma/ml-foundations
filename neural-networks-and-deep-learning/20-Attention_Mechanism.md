# 20. Attention Mechanism

## 1. Why Attention Was Needed

RNNs, LSTMs, and GRUs process sequential information by carrying state through time.

A simplified recurrent flow is:

```text
x₁ → h₁ → h₂ → h₃ → ... → hₜ
```

For a sequence-to-sequence model, an encoder may process an entire input sequence and produce a representation that the decoder uses to generate output.

The problem is that information from a long sequence can become difficult to preserve in a single fixed representation.

Conceptually:

```text
Long Input Sequence
        ↓
Compress Information
        ↓
Fixed Representation
        ↓
Generate Output
```

Even LSTM and GRU, although much better than basic RNNs at preserving long-term information, still face limitations when sequences become long and complex.

Attention introduced a different idea:

```text
Instead of forcing the model
to remember everything equally,

let it directly look at
the most relevant information
when it needs it.
```

---

## 2. The Central Idea of Attention

Suppose the model is processing:

```text
"The animal didn't cross the street because it was too tired."
```

To understand:

```text
it
```

the model should pay particular attention to:

```text
animal
```

rather than treating every previous word as equally important.

Conceptually:

```text
Current Item
     ↓
Look at Other Relevant Items
     ↓
Assign Importance
     ↓
Combine Important Information
     ↓
Contextual Representation
```

This is the basic idea of **attention**.

---

## 3. Attention as Selective Focus

Human intuition provides a useful analogy.

When reading:

```text
"The red car parked beside the old building was stolen."
```

while interpreting:

```text
stolen
```

we do not mentally treat every word as equally relevant.

Some words matter more:

```text
car
→ highly relevant

red
→ somewhat relevant

building
→ less relevant
```

Attention gives a neural network a learned mechanism for making similar relevance decisions.

---

## 4. Attention Does Not Mean Hard Selection

Attention usually does not choose exactly one item and discard everything else.

Instead, it assigns different weights.

For example:

```text
Token A → 0.05
Token B → 0.10
Token C → 0.70
Token D → 0.15
```

Then information from the tokens is combined according to these weights.

Thus attention is usually:

```text
Soft Selection
```

rather than:

```text
Hard Selection
```

This makes the operation differentiable and trainable using gradient-based learning.

---

## 5. The Three Core Components

Modern attention is commonly expressed using three concepts:

```text
Query
Key
Value
```

usually abbreviated:

```text
Q
K
V
```

These are the three most important terms to understand before studying transformers.

---

## 6. Query

The **query** represents:

```text
What am I looking for?
```

Suppose the model is currently processing a word and needs relevant context.

The query represents the information needs of that current position.

Conceptually:

```text
Query
→ "What information would be useful to me?"
```

---

## 7. Key

A **key** represents:

```text
What information does this item contain
that might make it relevant?
```

Each candidate item has a key.

The query is compared with the keys to determine relevance.

Conceptually:

```text
Query
   ↓
Compare With Keys
   ↓
Which Items Match What I Need?
```

---

## 8. Value

The **value** represents the actual information that will be retrieved if an item receives attention.

Thus:

```text
Query
→ What am I looking for?

Key
→ What does this item advertise?

Value
→ What information does this item provide?
```

This distinction is fundamental.

---

## 9. A Database Analogy

A useful intuition is to imagine a lookup system.

Suppose:

```text
Query
→ "I need information about neural networks."
```

Several records contain keys:

```text
Key 1 → Cooking
Key 2 → Neural Networks
Key 3 → Railway Engineering
Key 4 → Machine Learning
```

The query matches:

```text
Key 2
```

strongly and:

```text
Key 4
```

somewhat strongly.

Their associated values then contribute more to the result.

Attention performs a learned numerical version of this process.

---

## 10. Query-Key Similarity

The first mathematical step is determining how strongly a query matches each key.

One common similarity measure is the dot product.

For query:

```math
q
```

and key:

```math
k
```

the score is:

```math
\text{score}(q,k)
=
q^\top k
```

A larger dot product generally indicates greater alignment between the two vectors.

Thus:

```text
Query + Key
     ↓
Dot Product
     ↓
Attention Score
```

---

## 11. Multiple Keys

Suppose a query is compared with three keys:

```math
q^\top k_1
```

```math
q^\top k_2
```

```math
q^\top k_3
```

This produces three scores:

```text
Score 1
Score 2
Score 3
```

These scores tell the model how relevant each corresponding item appears to be.

---

## 12. Raw Scores Are Not Yet Attention Weights

Dot-product scores may contain arbitrary values such as:

```text
2.7
-0.4
5.1
1.8
```

These are difficult to interpret directly as proportions.

We therefore usually transform them using **softmax**.

If the scores are:

```math
s_1,s_2,\ldots,s_n
```

then:

```math
\alpha_i
=
\frac{e^{s_i}}
{\sum_j e^{s_j}}
```

The resulting attention weights satisfy:

```math
0<\alpha_i<1
```

and:

```math
\sum_i \alpha_i=1
```

---

## 13. Attention Weights

After softmax, the scores might become:

```text
Token 1 → 0.05
Token 2 → 0.15
Token 3 → 0.70
Token 4 → 0.10
```

These are the **attention weights**.

They answer:

```text
How much attention should
the current query give
to each item?
```

The third token receives the greatest influence.

---

## 14. Weighted Sum of Values

The attention weights are then applied to the values.

If:

```math
v_1,v_2,\ldots,v_n
```

are the value vectors, the attention output is:

```math
\text{Attention}
=
\sum_i
\alpha_i v_i
```

Thus:

```text
Attention Weight
        ×
Value
        ↓
Weighted Information
```

The weighted values are summed to produce the final contextual representation.

---

## 15. The Complete Basic Attention Process

The entire process can be summarized as:

```text
Query
  ↓
Compare With Keys
  ↓
Similarity Scores
  ↓
Softmax
  ↓
Attention Weights
  ↓
Multiply Values
  ↓
Weighted Sum
  ↓
Attention Output
```

Or more compactly:

```text
Q + K
→ Relevance

Relevance + V
→ Context
```

This is the central mechanism.

---

## 16. Matrix Form

When multiple queries, keys, and values are processed together, they can be represented as matrices:

```math
Q
```

```math
K
```

```math
V
```

Query-key scores are calculated using:

```math
QK^\top
```

Then softmax converts these scores into attention weights.

The result is multiplied by:

```math
V
```

Conceptually:

```math
QK^\top
→ Who Is Relevant?

Softmax
→ How Much?

Multiply by V
→ Retrieve and Combine Information
```

---

## 17. Scaled Dot-Product Attention

Transformers commonly use **scaled dot-product attention**.

The formula is:

```math
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}
{\sqrt{d_k}}
\right)V
```

where:

```math
d_k
```

is the dimensionality of the key vectors.

This equation is one of the central equations behind transformer architectures.

---

## 18. Why Divide by the Square Root of the Key Dimension?

As vector dimensionality increases, dot products can become large in magnitude.

Large scores fed into softmax can produce extremely sharp probabilities.

For example:

```text
Very Large Scores
      ↓
Softmax Saturation
      ↓
One Weight ≈ 1
Others ≈ 0
      ↓
Very Small Gradients
```

Scaling by:

```math
\sqrt{d_k}
```

keeps score magnitudes more manageable.

Thus:

```text
Scaling
→ More Stable Softmax
→ Better Gradient Behaviour
```

---

## 19. Where Do Q, K, and V Come From?

The query, key, and value vectors are not normally the raw input vectors themselves.

Suppose an input representation is:

```math
x
```

The model learns projection matrices:

```math
W_Q
```

```math
W_K
```

```math
W_V
```

and computes:

```math
q=xW_Q
```

```math
k=xW_K
```

```math
v=xW_V
```

These matrices are trainable parameters.

---

## 20. Why Separate Q, K, and V Projections?

The same input item may need to play different roles.

The model may need to learn:

```text
What should I search for?
→ Query Representation

How should I advertise my relevance?
→ Key Representation

What information should I contribute?
→ Value Representation
```

Separate learned projections allow these roles to develop independently.

---

## 21. Attention Is Learned

The model is not manually told:

```text
"Pay attention to nouns."

"Ignore adjectives."

"Look five words backward."
```

Instead:

```text
Training Data
      ↓
Loss
      ↓
Backpropagation
      ↓
Update WQ, WK, WV
      ↓
Useful Attention Patterns Emerge
```

Attention is therefore a learned mechanism.

---

## 22. Attention in Sequence-to-Sequence Models

Attention originally became especially important in encoder-decoder sequence models.

Without attention:

```text
Input Sequence
      ↓
Encoder
      ↓
Single Representation
      ↓
Decoder
      ↓
Output Sequence
```

The single representation can become an information bottleneck.

With attention:

```text
Encoder States
h₁  h₂  h₃  h₄  h₅
 ↑   ↑   ↑   ↑   ↑
 └── Decoder Can Attend ──┘
            ↓
       Current Output
```

The decoder can directly consult relevant encoder states while generating each output.

---

## 23. Attention Removes the Single-Vector Bottleneck

Consider machine translation.

The input might contain a long sentence.

Without attention:

```text
Entire Sentence
      ↓
Compress Everything
Into One Representation
      ↓
Decode Translation
```

With attention:

```text
Entire Sentence
      ↓
Multiple Encoder States
      ↓
Decoder Looks Back
at Relevant States
for Each Output
```

The decoder no longer needs every detail to survive inside one fixed vector.

---

## 24. Different Outputs Can Attend to Different Inputs

Suppose the input is:

```text
The black dog runs quickly
```

While generating one output word, the decoder may focus heavily on:

```text
dog
```

At another step, it may focus on:

```text
quickly
```

Thus:

```text
Output Step 1
→ Attention Pattern A

Output Step 2
→ Attention Pattern B

Output Step 3
→ Attention Pattern C
```

Attention is dynamically recomputed according to the current query.

---

## 25. Attention and Alignment

In sequence-to-sequence models, attention weights can represent a form of alignment.

For translation:

```text
Input Words
↕
Output Words
```

the model can learn which input positions are particularly relevant when producing each output position.

This is one reason attention became so powerful for translation tasks.

---

## 26. Self-Attention

The next major step is **self-attention**.

In ordinary encoder-decoder attention, a decoder query may attend to representations produced by an encoder.

In self-attention:

```text
Queries
Keys
Values
```

all come from the **same sequence**.

Conceptually:

```text
Sequence
   ↓
Every Position
Looks at Other Positions
in the Same Sequence
   ↓
Contextualized Sequence
```

This mechanism forms the foundation of transformers.

---

## 27. Why Is It Called Self-Attention?

Suppose the sequence is:

```text
The animal was tired because it had been running
```

When processing:

```text
it
```

the model can examine other positions in the same sequence.

Therefore:

```text
Sequence
→ Attends to Itself
```

Hence:

```text
Self-Attention
```

---

## 28. Self-Attention Creates Contextual Representations

Consider the word:

```text
bank
```

in:

```text
I deposited money at the bank
```

and:

```text
We sat on the river bank
```

The initial representation of:

```text
bank
```

may be similar in both cases.

After self-attention, its representation can incorporate surrounding context.

Thus:

```text
Same Token
+
Different Context
      ↓
Different Contextual Representation
```

This is enormously important in modern language modelling.

---

## 29. Self-Attention Step by Step

Suppose the sequence contains:

```text
A B C
```

Each token produces:

```text
Query
Key
Value
```

For token `A`:

```text
Query A
   ↓
Compare With:
Key A
Key B
Key C
   ↓
Attention Weights
   ↓
Weighted Combination of:
Value A
Value B
Value C
   ↓
Contextual Representation of A
```

The same process occurs for `B` and `C`.

---

## 30. A Token Can Attend to Itself

Self-attention does not necessarily mean:

```text
Only Look at Other Tokens
```

A token may also attend to itself.

For example:

```text
A attends to:
A
B
C
D
```

The learned weights determine how much influence each position receives.

---

## 31. Attention Matrix

For a sequence of four tokens, self-attention produces relationships such as:

```text
        K₁   K₂   K₃   K₄

Q₁      •    •    •    •
Q₂      •    •    •    •
Q₃      •    •    •    •
Q₄      •    •    •    •
```

Each row represents:

```text
One Query
```

Each column represents:

```text
One Key
```

Each cell represents:

```text
How Much That Query
Attends to That Key
```

After softmax, each row forms a distribution of attention weights.

---

## 32. Long-Range Relationships

A major advantage of self-attention is that distant positions can interact directly.

In an RNN:

```text
Token 1
→ Token 2
→ Token 3
→ ...
→ Token 100
```

information between tokens 1 and 100 must travel through many recurrent steps.

With self-attention:

```text
Token 1
────────────→ Token 100
```

the relationship can be represented directly in one attention operation.

---

## 33. Path Length Comparison

Suppose two relevant tokens are far apart.

### 33.1. RNN

```text
A → h₂ → h₃ → h₄ → ... → B
```

The path grows with sequence distance.

### 33.2. Self-Attention

```text
A ───────────────→ B
```

The interaction can occur directly.

This shorter information path helps models learn long-range dependencies.

---

## 34. Attention and Parallelism

RNNs have an inherent sequential dependency:

```math
h_t
```

requires:

```math
h_{t-1}
```

Therefore:

```text
Step 1
→ Step 2
→ Step 3
→ Step 4
```

must largely occur sequentially.

Self-attention does not require this recurrent chain.

During training, many sequence positions can therefore be processed simultaneously.

Conceptually:

```text
RNN
→ Sequential Computation

Self-Attention
→ Much Greater Parallelism
```

This became a major advantage for training large models.

---

## 35. Attention Has a Cost

Self-attention compares sequence positions with one another.

For sequence length:

```math
n
```

the attention score matrix contains approximately:

```math
n^2
```

relationships.

Thus standard self-attention has quadratic complexity with respect to sequence length:

```math
O(n^2)
```

Long sequences can therefore require substantial memory and computation.

Attention solves some problems while introducing new engineering challenges.

---

## 36. Attention Does Not Naturally Know Order

An RNN processes:

```text
A → B → C
```

sequentially.

Order is therefore built into the architecture.

Pure self-attention, however, processes relationships between representations without inherently knowing their sequence positions.

Without additional positional information:

```text
A B C
```

and a rearrangement of those items do not carry sufficient order information merely from the attention mechanism itself.

Transformers therefore introduce **positional information**.

This will become important in the next chapter.

---

## 37. Masked Attention

Sometimes a model must not see certain positions.

Consider language generation.

While predicting token:

```text
t
```

the model must not inspect future tokens:

```text
t+1
t+2
t+3
```

Otherwise, training would leak the answer.

A **causal mask** prevents attention to future positions.

Conceptually:

```text
Token 1
→ Can See Token 1

Token 2
→ Can See Tokens 1–2

Token 3
→ Can See Tokens 1–3

Token 4
→ Can See Tokens 1–4
```

Future information is blocked.

---

## 38. Causal Attention Matrix

A causal mask conceptually looks like:

```text
        K₁   K₂   K₃   K₄

Q₁      ✓    ✗    ✗    ✗
Q₂      ✓    ✓    ✗    ✗
Q₃      ✓    ✓    ✓    ✗
Q₄      ✓    ✓    ✓    ✓
```

This ensures:

```text
Past and Present
→ Visible

Future
→ Hidden
```

Causal attention is fundamental to autoregressive language models.

---

## 39. Padding Masks

Sequences within a batch may have different lengths.

For example:

```text
A B C D

A B <PAD> <PAD>
```

The model should not treat:

```text
<PAD>
```

as meaningful information.

A padding mask prevents attention from being assigned to padded positions.

Thus attention masks can serve different purposes:

```text
Causal Mask
→ Hide Future

Padding Mask
→ Hide Padding
```

---

## 40. Cross-Attention

Another important form is **cross-attention**.

Here:

```text
Queries
```

come from one sequence or representation, while:

```text
Keys and Values
```

come from another.

Conceptually:

```text
Sequence A
→ Queries

Sequence B
→ Keys + Values

        ↓
Cross-Attention
        ↓
A Retrieves Relevant
Information From B
```

Encoder-decoder transformers use this mechanism.

---

## 41. Self-Attention vs Cross-Attention

### 41.1. Self-Attention

```text
Q
K
V
```

come from the same sequence.

```text
Sequence
→ Looks at Itself
```

### 41.2. Cross-Attention

```text
Q
```

comes from one representation while:

```text
K and V
```

come from another.

```text
One Sequence
→ Looks at Another
```

This distinction becomes important in transformer architectures.

---

## 42. Multi-Head Attention

So far, we have described one attention operation.

Modern transformers typically use **multi-head attention**.

Instead of learning one set of attention relationships:

```text
One Attention Head
```

the model learns several:

```text
Head 1
Head 2
Head 3
...
Head h
```

Each head has its own learned projections.

---

## 43. Why Multiple Attention Heads?

Different relationships may matter simultaneously.

In language, one head might learn patterns related to:

```text
Subject ↔ Verb
```

another might capture:

```text
Pronoun ↔ Referent
```

another might focus on:

```text
Nearby Context
```

another might capture:

```text
Long-Range Dependency
```

These interpretations are only intuitive examples; individual heads do not necessarily map neatly onto human-defined linguistic roles.

The important point is:

```text
Multiple Heads
→ Multiple Learned Relationship Spaces
```

---

## 44. Multi-Head Attention Process

Each head computes its own:

```text
Q
K
V
```

projections.

Conceptually:

```text
Input
 ├→ Head 1 → Attention
 ├→ Head 2 → Attention
 ├→ Head 3 → Attention
 └→ Head 4 → Attention
              ↓
         Concatenate
              ↓
      Output Projection
              ↓
          Final Output
```

The heads are combined to form a richer representation.

---

## 45. Multi-Head Attention Mathematically

Each head can be written as:

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

The heads are concatenated:

```math
\text{MultiHead}(Q,K,V)
=
\text{Concat}
\left(
\text{head}_1,
\ldots,
\text{head}_h
\right)
W^O
```

where:

```math
W_i^Q,\;
W_i^K,\;
W_i^V
```

and:

```math
W^O
```

are learned projection matrices.

---

## 46. Attention Is Not Memory in the RNN Sense

An RNN carries information through:

```text
Hidden State
```

Attention works differently.

It allows a position to retrieve information from a collection of representations according to learned relevance.

Thus:

```text
RNN
→ Carry Context Forward

Attention
→ Retrieve Relevant Context Directly
```

This is an important conceptual distinction.

---

## 47. RNN vs Attention

| RNN | Attention |
|---|---|
| Information passed sequentially | Information retrieved by relevance |
| Hidden state carries context | Weighted combinations create context |
| Long paths between distant tokens | Direct interaction possible |
| Sequential computation | Highly parallelizable |
| Order naturally represented | Positional information required |
| Cost grows roughly linearly through sequence | Standard self-attention has quadratic pairwise cost |

Attention changes the fundamental way sequence relationships are represented.

---

## 48. LSTM vs Attention

LSTM improves recurrent memory using:

```text
Gates
+
Cell State
```

Attention instead asks:

```text
Why carry every important detail
through every intermediate step
if I can directly retrieve
the relevant representation?
```

Thus:

```text
LSTM
→ Better Memory Transport

Attention
→ Direct Information Access
```

This is one of the most important conceptual transitions in modern deep learning.

---

## 49. Attention Scores Are Context Dependent

The same word can receive different attention depending on the query.

Suppose:

```text
The dog chased the ball because it moved.
```

Different positions may assign different relevance to:

```text
dog
ball
it
moved
```

Attention is not a permanent importance score attached to each token.

Instead:

```text
Importance
=
Relative to the Current Query
```

This is essential.

---

## 50. Attention Is Not an Explanation Guarantee

Attention weights can sometimes provide useful clues about model behaviour.

However:

```text
High Attention Weight
```

should not automatically be interpreted as:

```text
Complete Explanation
of Why the Model Predicted Something
```

Neural networks contain many interacting transformations.

Attention weights show information-routing patterns, but they are not guaranteed to provide a complete causal explanation of model decisions.

---

## 51. From Attention to Transformers

Attention was initially used alongside recurrent architectures.

The major conceptual leap was:

```text
What if recurrence
is not required at all?
```

Instead of:

```text
RNN
+
Attention
```

a model could rely primarily on:

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

This leads to the **Transformer** architecture.

---

## 52. Why Attention Changed Deep Learning

Attention addressed several important limitations of recurrent sequence modelling.

It enabled:

```text
Direct Long-Range Interaction

Dynamic Context Selection

Greater Training Parallelism

Rich Contextual Representations

Multiple Simultaneous Relationship Patterns
```

These properties made attention particularly suitable for large-scale sequence modelling.

---

## 53. The Attention Equation to Remember

The central equation is:

```math
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}
{\sqrt{d_k}}
\right)V
```

Do not treat it as an isolated formula.

Read it from left to right:

```text
QKᵀ
→ Compare Queries With Keys

Divide by √dₖ
→ Stabilize Score Magnitudes

Softmax
→ Convert Scores to Attention Weights

Multiply by V
→ Retrieve Weighted Information
```

The equation is simply the mathematical expression of the attention idea.

---

## 54. The Q-K-V Mental Model

The most useful interpretation is:

```text
Query
→ What am I looking for?

Key
→ What do I contain?

Query × Key
→ How relevant are you to me?

Softmax
→ How much attention should I give you?

Value
→ What information will you give me?

Weighted Values
→ Context I actually receive
```

If this mental model is clear, the mathematics becomes much easier to remember.

---

## 55. Key Takeaways

- Attention allows a model to dynamically focus on information relevant to the current computation.
- Attention usually performs soft weighting rather than hard selection.
- Query, key, and value are the fundamental components of modern attention.
- The query represents what information is being sought.
- Keys represent how candidate items can be matched for relevance.
- Values contain the information retrieved from those items.
- Query-key similarity produces attention scores.
- Softmax converts attention scores into normalized attention weights.
- Attention output is a weighted combination of value vectors.
- Scaled dot-product attention is:

```math
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}
{\sqrt{d_k}}
\right)V
```

- Scaling prevents large dot products from causing excessively sharp softmax distributions.
- Q, K, and V are commonly produced using learned projection matrices.
- Attention helped remove the fixed-vector bottleneck of early encoder-decoder RNN models.
- Self-attention allows positions within the same sequence to interact directly.
- Self-attention produces contextual representations.
- Distant sequence positions can interact through a short computational path.
- Self-attention allows substantially more parallelism than recurrent processing.
- Standard self-attention has quadratic cost with sequence length.
- Self-attention does not inherently encode sequence order.
- Positional information is therefore required in transformer architectures.
- Causal masking prevents a position from seeing future information.
- Padding masks prevent padded positions from influencing attention.
- Cross-attention allows one representation to retrieve information from another.
- Multi-head attention learns several attention relationship spaces simultaneously.
- Attention is not equivalent to recurrent memory.
- Attention weights are query-dependent rather than fixed token-importance scores.
- Attention weights should not automatically be treated as complete explanations of model behaviour.
- Self-attention became the central mechanism underlying transformers.

### 55.1. Memory Hook

```text
ATTENTION:

"Don't carry everything
through every step.

Look directly at
what matters."


Q — Query
→ What am I looking for?

K — Key
→ What do you contain?

V — Value
→ What information
  will you give me?


Q × K
→ Relevance Score

Softmax
→ Attention Weight

Attention Weight × V
→ Retrieved Information


Core Equation:

Attention(Q,K,V)

= softmax(
    QKᵀ / √dₖ
  )V


Self-Attention:

Q, K, V
come from
the same sequence.


Cross-Attention:

Q
comes from one source

K, V
come from another.


Multi-Head Attention:

Don't learn
only one kind
of relationship.

Learn several
in parallel.


RNN:

Carry information
step by step.

LSTM:

Carry it better
using gates.

Attention:

Retrieve it directly.


The bridge is now complete:

RNN
 ↓
LSTM / GRU
 ↓
Attention
 ↓
Transformer
```
