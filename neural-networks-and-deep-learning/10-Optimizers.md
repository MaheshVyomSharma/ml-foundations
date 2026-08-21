# 10. Optimizers

## 1. What Is an Optimizer?

An **optimizer** is the algorithm that uses gradients computed during backpropagation to update a neural network's trainable parameters.

Backpropagation computes:

```math
\nabla_{\theta} J
```

The optimizer decides how this gradient should be used to modify the parameters:

```math
\theta
```

The relationship is:

```text
Forward Propagation
        ↓
Loss
        ↓
Backpropagation
        ↓
Gradients
        ↓
Optimizer
        ↓
Updated Parameters
```

Therefore:

```text
Backpropagation
= computes gradients

Optimizer
= uses gradients to update parameters
```

---

## 2. The Basic Gradient Descent Update

The simplest update rule is gradient descent:

```math
\theta_{t+1}
=
\theta_t
-
\eta \nabla_{\theta} J(\theta_t)
```

where:

```math
\eta
```

is the **learning rate**.

For an individual weight:

```math
w_{t+1}
=
w_t
-
\eta
\frac{\partial J}{\partial w_t}
```

The gradient determines the direction of steepest increase in loss.

Subtracting it moves the parameter toward lower loss.

Modern optimizers build upon this fundamental principle.

---

## 3. Why Do We Need More Sophisticated Optimizers?

Plain gradient descent can encounter several difficulties.

A neural-network loss landscape may contain:

- steep directions
- shallow directions
- noisy gradients
- plateaus
- narrow valleys
- saddle points

For example, imagine a narrow valley:

```text
\             /
 \           /
  \         /
   \       /
    \_____/
       ↓
   Minimum
```

Gradient descent may repeatedly oscillate across the steep sides while progressing slowly along the valley.

More advanced optimizers attempt to make this movement faster and more stable.

---

## 4. Stochastic Gradient Descent

In deep learning, **Stochastic Gradient Descent (SGD)** usually operates using mini-batches.

For a mini-batch `B`, the loss is:

```math
J_B
=
\frac{1}{|B|}
\sum_{i \in B}
L^{(i)}
```

The update becomes:

```math
\theta
\leftarrow
\theta
-
\eta \nabla_{\theta} J_B
```

Because the mini-batch contains only part of the dataset, its gradient is an estimate of the gradient over the entire training set.

This introduces some randomness or **noise** into the optimization process.

---

## 5. Gradient Noise

Different mini-batches may produce slightly different gradients:

```math
\nabla J_{B_1}
\neq
\nabla J_{B_2}
```

Therefore, the optimization path may fluctuate:

```text
Start
  ↘
   ↗
    ↘
     ↗
      ↘
       → Lower Loss
```

This noise is not necessarily harmful.

It can sometimes help optimization move through flat regions or escape certain undesirable regions of the loss landscape.

---

## 6. Limitation of Plain SGD

Suppose optimization occurs inside a narrow valley.

Plain SGD may move like:

```text
\           /
 \ ↘     ↙ /
  \ ↗   ↘ /
   \ ↘ ↙ /
    \ ↓ /
     \_/
```

It repeatedly moves from one side to another.

This can cause:

- oscillation
- slow convergence
- sensitivity to learning rate

One important solution is **momentum**.

---

## 7. Momentum

Momentum allows the optimizer to retain information about previous movement.

Instead of responding only to the current gradient, it maintains a running direction.

A simplified formulation is:

```math
v_t
=
\beta v_{t-1}
+
g_t
```

where:

```math
g_t
=
\nabla_{\theta} J(\theta_t)
```

The parameter update becomes:

```math
\theta_{t+1}
=
\theta_t
-
\eta v_t
```

The quantity:

```math
v_t
```

represents accumulated gradient information.

The hyperparameter:

```math
\beta
```

controls how strongly previous movement is retained.

---

## 8. Intuition Behind Momentum

Think of a ball rolling downhill.

Without momentum:

```text
Current Slope
     ↓
Current Movement
```

With momentum:

```text
Previous Movement
        +
Current Gradient
        ↓
New Movement
```

If gradients repeatedly point in approximately the same direction:

```text
Consistent Direction
        ↓
Momentum Builds
        ↓
Faster Movement
```

If gradients repeatedly alternate direction:

```text
Alternating Gradients
        ↓
Partial Cancellation
        ↓
Reduced Oscillation
```

Momentum therefore provides a form of short-term memory to the optimizer.

---

## 9. Why Momentum Helps

Consider again a narrow valley.

Without momentum:

```text
↘ ↙ ↘ ↙ ↘ ↙
```

With momentum:

```text
↘
 ↘
  ↓
  ↓
  ↓
```

Momentum suppresses some sideways oscillation while accelerating movement in directions where gradients remain consistent.

This often allows SGD to converge considerably faster.

---

## 10. Momentum Coefficient

The momentum coefficient is commonly represented by:

```math
\beta
```

A frequently used value is:

```math
\beta = 0.9
```

Conceptually:

```text
Small β
→ weak memory
→ current gradient dominates

Large β
→ stronger memory
→ previous direction matters more
```

The exact value is a hyperparameter.

---

## 11. Adaptive Learning Rates

Plain SGD normally uses the same global learning rate for every parameter:

```math
\theta_i
\leftarrow
\theta_i
-
\eta
\frac{\partial J}{\partial \theta_i}
```

However, different parameters may benefit from different effective step sizes.

This motivates **adaptive optimizers**.

Adaptive optimizers modify the effective learning rate for individual parameters based on their gradient history.

---

## 12. AdaGrad

**AdaGrad**, or Adaptive Gradient, accumulates squared gradients.

Let:

```math
g_t
=
\nabla_{\theta} J(\theta_t)
```

AdaGrad maintains:

```math
G_t
=
G_{t-1}
+
g_t^2
```

The update is approximately:

```math
\theta_{t+1}
=
\theta_t
-
\frac{\eta}
{\sqrt{G_t}+\epsilon}
g_t
```

where:

```math
\epsilon
```

is a small constant that prevents division by zero and improves numerical stability.

---

## 13. AdaGrad Intuition

Suppose a parameter repeatedly receives large gradients.

Then:

```math
G_t
```

becomes large.

Therefore:

```math
\frac{\eta}{\sqrt{G_t}+\epsilon}
```

becomes smaller.

Thus:

```text
Frequently Large Gradients
        ↓
Smaller Future Steps

Small or Sparse Gradients
        ↓
Relatively Larger Steps
```

AdaGrad can therefore work particularly well when gradients are sparse.

---

## 14. AdaGrad's Main Limitation

The accumulated quantity:

```math
G_t
```

can only increase.

As training continues:

```math
G_t \uparrow
```

which causes:

```math
\frac{\eta}{\sqrt{G_t}+\epsilon}
\downarrow
```

Eventually the effective learning rate may become extremely small.

Training can therefore slow dramatically or almost stop.

This is the main limitation of AdaGrad.

---

## 15. RMSProp

**RMSProp** addresses AdaGrad's continually accumulating denominator.

Instead of storing all past squared gradients equally, RMSProp maintains an exponentially weighted moving average:

```math
s_t
=
\beta s_{t-1}
+
(1-\beta)g_t^2
```

The parameter update is:

```math
\theta_{t+1}
=
\theta_t
-
\frac{\eta}
{\sqrt{s_t}+\epsilon}
g_t
```

Older gradients gradually lose influence.

---

## 16. RMSProp Intuition

RMSProp approximately asks:

> How large have this parameter's gradients been recently?

If recent gradients are consistently large:

```text
Effective Step Size
        ↓
```

If recent gradients are relatively small:

```text
Effective Step Size
        ↑
```

RMSProp therefore adapts the learning rate without allowing the historical squared-gradient total to grow forever.

---

## 17. Adam

**Adam** stands for **Adaptive Moment Estimation**.

It combines ideas from:

```text
Momentum
+
RMSProp
```

Adam tracks two quantities:

```text
First Moment
→ moving average of gradients

Second Moment
→ moving average of squared gradients
```

This provides both:

```text
Direction Memory
+
Adaptive Step Scaling
```

Adam is one of the most widely used optimizers in deep learning.

---

## 18. Adam's First Moment

Adam maintains a moving average of the gradients:

```math
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
```

This resembles momentum.

It estimates the general direction in which gradients have recently been pointing.

Conceptually:

```text
First Moment
≈ Direction Memory
```

---

## 19. Adam's Second Moment

Adam also maintains a moving average of squared gradients:

```math
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
```

This tracks recent gradient magnitude.

Conceptually:

```text
First Moment
→ Which direction?

Second Moment
→ How large?
```

The two estimates are then combined to determine the parameter update.

---

## 20. Bias Correction in Adam

Adam usually initializes:

```math
m_0=0
```

and:

```math
v_0=0
```

During the first few training steps, these moving averages are therefore biased toward zero.

Adam corrects this using:

```math
\hat{m}_t
=
\frac{m_t}
{1-\beta_1^t}
```

and:

```math
\hat{v}_t
=
\frac{v_t}
{1-\beta_2^t}
```

These are called **bias-corrected moment estimates**.

---

## 21. Adam Parameter Update

Adam updates the parameters using:

```math
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
```

The numerator provides directional information.

The denominator scales the update according to recent gradient magnitude.

Conceptually:

```text
Gradient Direction History
          +
Gradient Magnitude History
          ↓
Adaptive Parameter Update
```

---

## 22. Common Adam Hyperparameters

Common starting values are:

```math
\beta_1=0.9
```

```math
\beta_2=0.999
```

and a small:

```math
\epsilon
```

for numerical stability.

A frequently used initial learning rate is:

```math
\eta=0.001
```

These are useful defaults, not universal laws.

---

## 23. SGD vs Adam

A simplified comparison is:

| Property | SGD | Adam |
|---|---|---|
| Uses current gradient | Yes | Yes |
| Uses gradient history | No, unless momentum is added | Yes |
| Adaptive step size | No | Yes |
| Additional memory | Low | Higher |
| Initial tuning | Often more sensitive | Often easier |
| Early convergence | Can be slower | Often faster |
| Generalization | Can be excellent | Can also be excellent |

Neither optimizer is universally superior.

The best choice depends on the problem.

---

## 24. Why Adam Is Popular

Adam is popular because it:

- adapts effective parameter step sizes
- incorporates momentum-like behaviour
- handles noisy gradients well
- often works reasonably with default settings
- frequently converges quickly

Therefore, Adam is often a strong starting point when developing a neural network.

However:

```text
Good General-Purpose Default
≠
Universally Best Optimizer
```

SGD with momentum remains highly effective in many applications.

---

## 25. Optimizer State

More advanced optimizers need to remember information from previous iterations.

Momentum stores quantities such as:

```math
v_t
```

Adam stores:

```math
m_t
```

and:

```math
v_t
```

for each parameter.

This additional information is called **optimizer state**.

Therefore, a training checkpoint may contain:

```text
Model Parameters
+
Optimizer State
+
Training Progress
```

Saving optimizer state allows training to resume consistently.

---

## 26. Adaptive Optimizers Still Need a Learning Rate

Adam and RMSProp adapt effective step sizes, but they still contain a base learning rate:

```math
\eta
```

Therefore:

```text
Adaptive Optimizer
≠
No Learning Rate
```

A poorly chosen learning rate can still make training:

- too slow
- unstable
- divergent

The optimizer and learning rate must work together.

---

## 27. Learning Rate Schedules

The learning rate does not have to remain constant throughout training.

Instead:

```math
\eta=\eta(t)
```

where the learning rate changes with training time.

A common strategy is:

```text
Early Training
→ Larger Steps
→ Rapid Progress

Later Training
→ Smaller Steps
→ Fine Adjustment
```

This is called a **learning rate schedule**.

---

## 28. Step Decay

A simple schedule reduces the learning rate at predetermined intervals.

For example:

```text
Epochs 1–10
→ 0.01

Epochs 11–20
→ 0.001

Epochs 21–30
→ 0.0001
```

Large early steps allow rapid movement through parameter space.

Smaller later steps allow finer optimization.

---

## 29. Exponential Decay

The learning rate can also decrease smoothly.

For example:

```math
\eta_t
=
\eta_0 e^{-kt}
```

where:

```math
\eta_0
```

is the initial learning rate and:

```math
k
```

controls the decay rate.

As training progresses:

```math
\eta_t \downarrow
```

---

## 30. Reduce Learning Rate on Plateau

Another strategy monitors validation performance.

Conceptually:

```text
Validation Loss Improving
        ↓
Keep Learning Rate

Validation Loss Stagnates
        ↓
Reduce Learning Rate
```

This allows the optimizer to take larger steps while useful progress is occurring and smaller steps when finer adjustments are needed.

---

## 31. Optimizer vs Learning Rate Scheduler

These concepts have different responsibilities.

```text
Optimizer
→ How gradients produce parameter updates

Learning Rate Scheduler
→ How the learning rate changes over time
```

For example:

```text
Optimizer
→ Adam

Initial Learning Rate
→ 0.001

Scheduler
→ Reduce learning rate when validation loss plateaus
```

They work together during training.

---

## 32. Weight Decay

An optimizer may also apply **weight decay**.

Conceptually, weight decay gradually pushes parameters toward smaller magnitudes.

A simplified update can be written as:

```math
W
\leftarrow
W
-
\eta
\left(
\nabla_W J
+
\lambda W
\right)
```

where:

```math
\lambda
```

controls the strength of the penalty.

Weight decay can help discourage excessively large weights and improve generalization.

It is closely related to L2 regularization, although the exact relationship depends on the optimizer.

---

## 33. AdamW

**AdamW** is a widely used variation of Adam.

It **decouples weight decay** from Adam's adaptive gradient update.

Conceptually:

```text
Adam
→ Adaptive Gradient Update

Weight Decay
→ Separate Parameter Shrinkage
```

This avoids mixing weight decay directly into Adam's adaptive gradient scaling.

AdamW is widely used in modern deep-learning systems.

---

## 34. Gradient Clipping

If gradients become extremely large, optimizer updates can become unstable.

**Gradient clipping** limits gradient magnitude before the optimizer uses it.

Conceptually:

```text
Huge Gradient
      ↓
Gradient Clipping
      ↓
Controlled Gradient
      ↓
Optimizer
      ↓
Parameter Update
```

One common method clips the gradient norm.

If:

```math
\|g\|>c
```

the gradient can be rescaled so that its norm becomes:

```math
\|g\|=c
```

where:

```math
c
```

is the clipping threshold.

Gradient clipping is especially useful when exploding gradients are a risk.

---

## 35. Optimizers Do Not Compute Gradients

The optimizer does not perform backpropagation.

Backpropagation calculates:

```math
g_t
=
\nabla_{\theta}J
```

The optimizer receives this gradient and determines how the parameters should change.

Therefore:

```text
Loss
 ↓
Backpropagation
 ↓
Gradient
 ↓
Optimizer
 ↓
Parameter Update
```

Keeping these responsibilities separate is important.

---

## 36. Choosing an Optimizer

A useful practical starting point is:

```text
Simple Baseline
→ SGD

Need Better Directional Stability
→ SGD + Momentum

Strong General-Purpose Default
→ Adam

Adam with Decoupled Weight Decay
→ AdamW
```

The final choice should be evaluated using validation performance.

Optimizer choice is a **hyperparameter decision**.

---

## 37. Optimizers in the Complete Training Loop

The complete training process is now:

```text
Mini-Batch
   ↓
Forward Propagation
   ↓
Prediction
   ↓
Loss
   ↓
Backpropagation
   ↓
Gradients
   ↓
Optimizer
   ├── SGD
   ├── SGD + Momentum
   ├── RMSProp
   ├── Adam
   └── AdamW
   ↓
Updated Parameters
   ↓
Next Mini-Batch
```

The architecture determines what the network can represent.

Backpropagation determines how the loss depends on its parameters.

The optimizer determines how the network moves through parameter space.

---

## 38. Key Takeaways

- An optimizer uses gradients to update trainable parameters.
- Backpropagation computes gradients; the optimizer uses them.
- SGD updates parameters using mini-batch gradients.
- Mini-batch gradients contain noise because they estimate the full-dataset gradient.
- Momentum incorporates information from previous gradients.
- Momentum can reduce oscillation and accelerate movement in consistent directions.
- AdaGrad adapts effective learning rates but may shrink them excessively over time.
- RMSProp uses a moving average of squared gradients to avoid AdaGrad's continual accumulation.
- Adam combines momentum-like first-moment estimation with RMSProp-like second-moment estimation.
- Adam uses bias correction during early training.
- Adam is a strong general-purpose optimizer but is not universally best.
- Advanced optimizers maintain additional optimizer state.
- Adaptive optimizers still require a base learning rate.
- Learning rate schedules modify the learning rate during training.
- Weight decay discourages excessively large weights.
- AdamW decouples weight decay from Adam's adaptive gradient update.
- Gradient clipping helps control exploding gradients.
- Optimizer choice is a hyperparameter decision.

### 38.1. Memory Hook

```text
Backpropagation
→ Compute Gradient

Optimizer
→ Use Gradient

SGD
→ Current Gradient

Momentum
→ Gradient + Memory

RMSProp
→ Adapt Using Recent Gradient Magnitude

Adam
→ Momentum + RMSProp Ideas

AdamW
→ Adam + Decoupled Weight Decay

Learning Rate
→ How Big a Step?

Scheduler
→ How Should Step Size Change?

Gradient Clipping
→ Prevent Huge Updates

Core Idea:

Gradient tells us WHERE.

Optimizer decides HOW to move.
```
