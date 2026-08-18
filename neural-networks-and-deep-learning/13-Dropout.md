# 13. Dropout

## 1.1 What Is Dropout?

**Dropout** is a regularization technique in which a random subset of neuron outputs is temporarily set to zero during training.

Suppose a hidden layer contains:

```text
○ ○ ○ ○ ○ ○
```

During one training step, dropout might produce:

```text
○ × ○ ○ × ○
```

where `×` represents neurons whose outputs have been dropped.

During another step:

```text
× ○ ○ × ○ ○
```

A different subset is removed.

This randomness prevents the network from becoming overly dependent on particular neurons or pathways.

---

## 1.2 Why Dropout Is Needed

A large neural network may contain enough capacity to memorize training-specific patterns.

Neurons can also become strongly dependent on one another.

For example:

```text
Neuron A
expects
Neuron B
to detect a particular feature
```

and:

```text
Neuron B
expects
Neuron C
to always be available
```

Such dependencies can contribute to overfitting.

Dropout disrupts these relationships during training.

Conceptually:

```text
Randomly Remove Neurons
        ↓
Network Cannot Rely on Fixed Paths
        ↓
Features Must Become More Robust
        ↓
Better Generalization
```

---

## 1.3 Dropout Probability

Dropout is controlled by a probability.

Let:

```math
p
```

represent the probability that a neuron's activation is dropped.

For example:

```math
p=0.5
```

means each activation has a 50% chance of being set to zero during a training step.

If:

```math
p=0.2
```

then approximately 20% of activations are dropped.

The remaining probability is:

```math
1-p
```

which is sometimes called the **keep probability**.

---

## 1.4 Dropout Mask

Suppose a hidden-layer activation vector is:

```math
\mathbf{a}
=
\begin{bmatrix}
a_1 \\
a_2 \\
a_3 \\
a_4
\end{bmatrix}
```

Dropout creates a random binary mask:

```math
\mathbf{m}
=
\begin{bmatrix}
1 \\
0 \\
1 \\
0
\end{bmatrix}
```

The masked activation becomes:

```math
\mathbf{a}_{\text{drop}}
=
\mathbf{m}
\odot
\mathbf{a}
```

where:

```math
\odot
```

denotes element-wise multiplication.

Therefore:

```math
\mathbf{a}_{\text{drop}}
=
\begin{bmatrix}
a_1 \\
0 \\
a_3 \\
0
\end{bmatrix}
```

The dropped neurons contribute nothing to the next layer during that training step.

---

## 1.5 Random Mask Generation

Each mask value can be viewed as a Bernoulli random variable.

For keep probability:

```math
q=1-p
```

we can write:

```math
m_i
\sim
\mathrm{Bernoulli}(q)
```

Thus:

```math
m_i
=
\begin{cases}
1, & \text{with probability } q \\
0, & \text{with probability } p
\end{cases}
```

A new mask is usually generated during each training step.

Therefore, the effective network changes continually during training.

---

## 1.6 Dropout Creates Many Effective Networks

Consider a network containing many neurons.

Each possible dropout mask produces a slightly different active subnetwork.

Conceptually:

```text
Training Step 1
→ Network A

Training Step 2
→ Network B

Training Step 3
→ Network C

Training Step 4
→ Network D
```

All these subnetworks share the same underlying parameters.

This gives dropout an interpretation similar to training many related models and combining their behaviour.

---

## 1.7 Ensemble Intuition

An **ensemble** combines predictions from multiple models to obtain more robust predictions.

Dropout has an ensemble-like effect because many different subnetworks are trained.

Conceptually:

```text
Subnetwork 1
Subnetwork 2
Subnetwork 3
Subnetwork 4
      ↓
Shared Parameters
      ↓
Combined Behaviour
```

At inference time, the full network approximates the combined behaviour of these many subnetworks.

This is one reason dropout can improve generalization.

---

## 1.8 Co-Adaptation

Without dropout, neurons may become highly specialized around the presence of specific other neurons.

This is called **co-adaptation**.

For example:

```text
Neuron A
→ learns only when Neuron B is active

Neuron B
→ depends heavily on Neuron C
```

Dropout interrupts these dependencies.

Because any neighbouring neuron may disappear temporarily, each neuron is encouraged to learn features that remain useful under many different combinations.

---

## 1.9 Training-Time Behaviour

Dropout is active during training.

Suppose:

```math
\mathbf{a}
```

is a layer's activation.

A random mask:

```math
\mathbf{m}
```

is applied:

```math
\mathbf{a}_{\text{drop}}
=
\mathbf{m}\odot\mathbf{a}
```

The next layer then receives the modified activations.

Therefore:

```text
Training
→ Dropout ON
```

The exact set of active neurons changes repeatedly.

---

## 1.10 Inference-Time Behaviour

During inference, dropout is normally disabled.

Therefore:

```text
Training
→ Dropout ON

Inference
→ Dropout OFF
```

The full network is used when predicting unseen data.

This distinction is essential.

If neurons continued to be randomly removed during ordinary inference, predictions would fluctuate unnecessarily.

---

## 1.11 Why Scaling Is Necessary

Suppose a neuron normally outputs:

```math
a=10
```

and the keep probability is:

```math
q=0.5
```

During training, the neuron is active only half the time.

Its expected contribution without compensation would therefore be:

```math
0.5(10)=5
```

But during inference, dropout is disabled and the output would become:

```math
10
```

The activation scale would therefore differ between training and inference.

This must be corrected.

---

## 1.12 Inverted Dropout

Modern frameworks usually use **inverted dropout**.

During training, surviving activations are divided by the keep probability:

```math
\mathbf{a}_{\text{drop}}
=
\frac{
\mathbf{m}\odot\mathbf{a}
}{
q
}
```

where:

```math
q=1-p
```

For:

```math
p=0.5
```

we have:

```math
q=0.5
```

so surviving activations are multiplied by:

```math
\frac{1}{0.5}=2
```

This preserves the expected activation magnitude.

---

## 1.13 Why Inverted Dropout Works

Suppose:

```math
a=10
```

and:

```math
q=0.5
```

During training:

- with probability `0.5`, activation becomes `0`
- with probability `0.5`, activation becomes:

```math
\frac{10}{0.5}=20
```

The expected value is:

```math
0.5(0)+0.5(20)=10
```

which matches the original activation.

Thus:

```math
E[a_{\text{drop}}]=a
```

This means inference can simply disable dropout without requiring additional scaling.

---

## 1.14 Dropout During Backpropagation

If a neuron is dropped during the forward pass:

```math
a_i=0
```

then it also receives no useful gradient contribution during that training step.

Conceptually:

```text
Neuron Dropped
     ↓
Forward Contribution = 0
     ↓
Backward Gradient = 0
     ↓
No Update Through That Path
```

On another training step, the same neuron may remain active and participate normally.

---

## 1.15 Dropout Rate

The **dropout rate** is the probability:

```math
p
```

of dropping an activation.

Typical values might include:

```math
0.1,\;0.2,\;0.3,\;0.5
```

but there is no universally correct value.

A larger dropout rate means stronger regularization.

Conceptually:

```text
Small p
→ Weak Dropout

Large p
→ Strong Dropout
```

---

## 1.16 Too Little Dropout

If:

```math
p
```

is very small, most neurons remain active almost all the time.

Then regularization may be weak.

If the network is heavily overfitting:

```text
Very Small Dropout
→ May Not Be Enough
```

Validation performance should determine whether stronger regularization is useful.

---

## 1.17 Too Much Dropout

If dropout is too strong:

```text
Too Many Neurons Removed
        ↓
Network Has Too Little Effective Capacity
        ↓
Training Becomes Difficult
        ↓
Underfitting
```

For example, dropping 90% of neurons in a small network would usually be excessive.

Thus:

```text
More Dropout
≠ Automatically Better
```

---

## 1.18 Dropout and Model Capacity

Dropout temporarily reduces the effective capacity of the network during each training step.

Suppose a layer contains 100 neurons with:

```math
p=0.5
```

On average, about:

```math
100(1-0.5)=50
```

neurons remain active during a particular training step.

However, the identity of those neurons changes continually.

The full architecture remains available across training.

---

## 1.19 Dropout Is a Hyperparameter

The dropout rate is not learned through backpropagation in the standard approach.

It is selected as a **hyperparameter**.

For example:

```text
Dense(128)
↓
ReLU
↓
Dropout(0.3)
↓
Dense(64)
↓
ReLU
↓
Dropout(0.2)
```

The values:

```math
0.3
```

and:

```math
0.2
```

are chosen during model design and tuning.

---

## 1.20 Where Is Dropout Applied?

Dropout is commonly applied after a hidden-layer activation.

Conceptually:

```text
Dense Layer
    ↓
Activation
    ↓
Dropout
    ↓
Next Layer
```

For example:

```text
Dense(128)
↓
ReLU
↓
Dropout
↓
Dense(64)
```

The dropout layer has no trainable weights of its own.

It simply modifies activations during training.

---

## 1.21 Dropout on the Output Layer

Dropout is generally **not applied to the final output layer**.

For example:

```text
Hidden Layer
↓
Dropout
↓
Output Layer
```

is normal.

But:

```text
Output Layer
↓
Dropout
```

would randomly remove final prediction components during training and usually makes little sense.

Dropout primarily regularizes intermediate representations.

---

## 1.22 Dropout on Input Features

Dropout can also be applied to inputs.

Conceptually:

```text
Input Features
      ↓
Random Feature Removal
      ↓
Network
```

This forces the network to avoid relying excessively on any single input feature.

However, input dropout is usually used more cautiously than hidden-layer dropout.

Whether it is useful depends strongly on the data and architecture.

---

## 1.23 Example Architecture

Consider:

```text
Input
 ↓
Dense(256)
 ↓
ReLU
 ↓
Dropout(0.5)
 ↓
Dense(128)
 ↓
ReLU
 ↓
Dropout(0.3)
 ↓
Dense(1)
 ↓
Sigmoid
```

During training:

```text
Dropout Layers
→ Active
```

During inference:

```text
Dropout Layers
→ Disabled
```

All neurons contribute to the final prediction.

---

## 1.24 Dropout and Training Loss

Dropout makes the training problem harder.

Therefore, training loss may be higher than it would be without dropout.

This is expected.

For example:

```text
Without Dropout
Training Accuracy = 99%
Validation Accuracy = 87%

With Dropout
Training Accuracy = 94%
Validation Accuracy = 92%
```

The second model is preferable because it generalizes better.

Thus:

```text
Worse Training Fit
can produce
Better Validation Performance
```

---

## 1.25 Dropout and Validation Loss

During validation, dropout is disabled.

Therefore, the model may sometimes perform better on validation data than expected relative to the noisy training process.

Training metrics are measured while neurons are being randomly removed.

Validation metrics are measured using the complete network.

This difference should be remembered when interpreting learning curves.

---

## 1.26 Dropout vs L2 Regularization

Both techniques reduce overfitting, but they work differently.

### L2 Regularization

Penalizes large weights:

```math
J
=
L_{\text{data}}
+
\lambda\|W\|^2
```

### Dropout

Randomly removes activations during training.

Conceptually:

```text
L2
→ constrain weight magnitude

Dropout
→ disrupt neuron dependencies
```

They can be used independently or together.

---

## 1.27 Dropout vs Early Stopping

Early stopping limits how long training continues.

Dropout changes the network during every training step.

Therefore:

```text
Early Stopping
→ Controls Training Duration

Dropout
→ Introduces Random Structural Noise
```

Both can act as regularization but through different mechanisms.

---

## 1.28 Dropout vs Data Augmentation

Data augmentation introduces variation in the **input data**.

Dropout introduces variation inside the **network**.

```text
Data Augmentation
→ Randomize Inputs

Dropout
→ Randomize Active Network Paths
```

Both make memorization more difficult.

---

## 1.29 Dropout and Batch Normalization

Dropout and batch normalization can both appear in the same network, but they serve different primary purposes.

```text
Dropout
→ Regularization

Batch Normalization
→ Stabilize / Normalize Layer Activations
```

Batch normalization may also have some regularizing effect.

In some modern architectures, strong normalization and large datasets reduce the need for heavy dropout.

The optimal combination remains architecture-dependent.

---

## 1.30 Dropout in Convolutional Networks

Dropout can be used in CNNs, especially in fully connected portions of a network.

However, ordinary dropout on individual spatial activations may be less effective in some convolutional layers because nearby activations are strongly correlated.

Variants such as:

```text
Spatial Dropout
```

can drop entire feature maps or channels rather than individual activations.

This preserves the same general idea:

```text
Remove Information During Training
→ Force More Robust Representations
```

---

## 1.31 Dropout in Recurrent Networks

Dropout can also be used with recurrent neural networks.

However, blindly applying independent random masks at every time step can disrupt temporal information.

Specialized recurrent dropout strategies may therefore use more structured masks.

The central regularization principle remains the same, but implementation details matter more in recurrent architectures.

---

## 1.32 Monte Carlo Dropout

Normally:

```text
Training
→ Dropout ON

Inference
→ Dropout OFF
```

However, dropout can deliberately remain active during inference in a technique called **Monte Carlo Dropout**.

The same input is passed through the network multiple times:

```text
Input
 ↓
Prediction 1

Input
 ↓
Prediction 2

Input
 ↓
Prediction 3
```

Because different dropout masks are sampled, predictions vary.

This variation can provide an approximate measure of predictive uncertainty.

This is an advanced use of dropout rather than its standard inference behaviour.

---

## 1.33 Dropout Does Not Reduce Stored Model Size

A common misunderstanding is:

```text
Dropout removes neurons
→ therefore model becomes permanently smaller
```

This is incorrect.

Dropout removes neurons **temporarily during individual training passes**.

The underlying parameters still exist.

Therefore:

```text
Dropout
≠ Network Pruning
```

Pruning permanently removes or disables parameters.

Dropout is temporary stochastic regularization.

---

## 1.34 Dropout Does Not Remove Parameters Permanently

Suppose a neuron is dropped during one mini-batch.

Its weights still remain part of the network.

During the next batch:

```text
Same Neuron
→ May Be Active Again
```

So:

```text
Dropped
≠ Deleted
```

This distinction is important.

---

## 1.35 Dropout and Randomness

Because dropout uses random masks, training becomes stochastic.

Two training runs can therefore follow different optimization paths even when everything else is identical.

Random seeds can improve reproducibility.

However, dropout's randomness is intentional:

```text
Randomness
→ Different Effective Networks
→ Reduced Co-Adaptation
→ Better Generalization Potential
```

---

## 1.36 When Should Dropout Be Considered?

Dropout is most useful when there is evidence of overfitting.

For example:

```text
Training Performance
→ Excellent

Validation Performance
→ Significantly Worse
```

Adding dropout may reduce this gap.

If both training and validation performance are poor, stronger dropout may make the situation worse because the network may already be underfitting.

---

## 1.37 Practical Decision Rule

A useful workflow is:

```text
Training Good?
│
├─ No
│   ↓
│ Underfitting
│ → Don't immediately add more dropout
│
└─ Yes
    ↓
Validation Much Worse?
    │
    ├─ No
    │ → Dropout may not be necessary
    │
    └─ Yes
      → Consider dropout / other regularization
```

Dropout should solve an observed generalization problem rather than be inserted automatically into every network.

---

## 1.38 Dropout in the Training Loop

With dropout, the training process becomes:

```text
Mini-Batch
   ↓
Forward Propagation
   ↓
Random Dropout Masks
   ↓
Prediction
   ↓
Loss
   ↓
Backpropagation Through Active Paths
   ↓
Optimizer
   ↓
Parameter Update
   ↓
New Mini-Batch
   ↓
New Dropout Masks
```

During inference:

```text
Input
 ↓
Full Network
 ↓
Prediction
```

No ordinary dropout mask is applied.

---

## 1.39 Key Takeaways

- Dropout is a neural-network regularization technique.
- During training, a random subset of activations is set to zero.
- The probability of dropping a neuron is the dropout rate `p`.
- The keep probability is:

```math
1-p
```

- A new dropout mask is typically sampled repeatedly during training.
- Dropout creates many different effective subnetworks that share parameters.
- It reduces excessive co-adaptation between neurons.
- Modern frameworks usually use inverted dropout.
- In inverted dropout, surviving activations are scaled by:

```math
\frac{1}{1-p}
```

- Dropout is active during training and disabled during ordinary inference.
- Dropped neurons do not participate in forward or backward computation for that training step.
- Dropout rate is a hyperparameter.
- Too little dropout may provide insufficient regularization.
- Too much dropout can cause underfitting.
- Dropout is typically applied to hidden layers rather than the output layer.
- Dropout may increase training loss while improving validation performance.
- Dropout, L2 regularization, early stopping, and data augmentation regularize models through different mechanisms.
- Dropout does not permanently remove neurons or reduce the stored network size.
- Monte Carlo dropout is an advanced technique that intentionally keeps dropout active during inference to estimate uncertainty.
- Dropout is most useful when a model is overfitting rather than underfitting.

### Memory Hook

```text
Dropout
= Randomly Disable Neurons During Training

Why?
→ Prevent Co-Adaptation
→ Force Robust Features
→ Reduce Overfitting

Training
→ Dropout ON

Inference
→ Dropout OFF

Drop Probability
→ p

Keep Probability
→ 1 - p

Inverted Dropout
→ Scale Survivors by 1 / (1 - p)

Dropped
≠ Deleted

Too Little Dropout
→ Weak Regularization

Too Much Dropout
→ Underfitting

Core Idea:

Don't let the network
depend too heavily
on any one path.
```
