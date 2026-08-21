# 15. Hyperparameters and Hyperparameter Tuning

## 1. What Is a Hyperparameter?

A **hyperparameter** is a configuration value chosen before or around the training process rather than learned directly from the training data through backpropagation.

Examples include:

- learning rate
- batch size
- number of hidden layers
- number of neurons per layer
- activation function
- optimizer
- dropout rate
- regularization strength
- number of epochs
- weight initialization strategy

These choices influence how the model is structured and how it learns.

Conceptually:

```text
Hyperparameters
      ↓
Control
      ↓
Architecture + Training Process
      ↓
Learned Model
```

---

## 2. Parameters vs Hyperparameters

This distinction is fundamental.

### 2.1. Parameters

Parameters are learned during training.

Examples:

```math
W
```

and:

```math
b
```

Backpropagation and the optimizer modify them.

### 2.2. Hyperparameters

Hyperparameters control the training process or model design.

Examples:

```math
\eta
```

for learning rate and:

```math
\lambda
```

for regularization strength.

Therefore:

```text
Parameters
→ Learned by the model

Hyperparameters
→ Chosen to control the model and training
```

---

## 3. A Concrete Example

Suppose we build:

```text
Input
 ↓
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
Output
```

The following are hyperparameters:

```text
Number of hidden layers = 2
Hidden units = 128, 64
Activation = ReLU
Dropout rate = 0.3
Optimizer = Adam
Learning rate = 0.001
Batch size = 32
```

But the actual values inside the weight matrices and bias vectors are parameters:

```math
W^{[1]},b^{[1]},W^{[2]},b^{[2]},W^{[3]},b^{[3]}
```

These are learned from data.

---

## 4. Parameters vs Optimizer State

Optimizer state is another concept that should remain separate.

For Adam, for example, the optimizer maintains quantities such as:

```math
m_t
```

and:

```math
v_t
```

These are neither ordinary model parameters nor manually selected hyperparameters.

They are internal state accumulated by the optimizer during training.

Thus:

```text
Model Parameters
→ Learned weights and biases

Hyperparameters
→ Training / architecture choices

Optimizer State
→ Internal history maintained during optimization
```

---

## 5. Why Hyperparameters Matter

The same dataset and same broad model type can behave very differently under different hyperparameter choices.

For example:

```text
Learning Rate Too Small
→ Very Slow Training

Learning Rate Too Large
→ Unstable Training

Network Too Small
→ Underfitting

Network Too Large
→ Greater Overfitting Risk

Dropout Too Strong
→ Underfitting

Dropout Too Weak
→ Insufficient Regularization
```

Hyperparameters therefore strongly influence both optimization and generalization.

---

## 6. Major Categories of Hyperparameters

Hyperparameters can be grouped into several categories.

### 6.1. Architecture Hyperparameters

```text
Number of Layers
Number of Neurons
Activation Functions
```

### 6.2. Optimization Hyperparameters

```text
Learning Rate
Optimizer
Momentum
Learning Rate Schedule
```

### 6.3. Data / Training Hyperparameters

```text
Batch Size
Number of Epochs
```

### 6.4. Regularization Hyperparameters

```text
Dropout Rate
Weight Decay
L1 / L2 Strength
```

The distinction is conceptual rather than absolute, but it helps organize the tuning process.

---

## 7. Learning Rate

The learning rate is one of the most important hyperparameters.

The basic update is:

```math
\theta_{t+1}
=
\theta_t
-
\eta\nabla_{\theta}J
```

where:

```math
\eta
```

is the learning rate.

It controls the size of the optimizer's steps through parameter space.

---

## 8. Learning Rate Too Small

If:

```math
\eta
```

is too small:

```text
Gradient
   ↓
Tiny Update
   ↓
Tiny Update
   ↓
Tiny Update
   ↓
Very Slow Progress
```

Training may eventually work, but require unnecessarily many iterations.

The loss curve may decrease very slowly.

---

## 9. Learning Rate Too Large

If the learning rate is too large:

```text
Move Toward Minimum
        ↓
Overshoot
        ↓
Overshoot Again
        ↓
Oscillation / Divergence
```

Instead of converging:

```math
J(\theta_t)\rightarrow\text{minimum}
```

the loss may fluctuate or increase.

Extremely large learning rates can make training numerically unstable.

---

## 10. A Useful Learning Rate Intuition

Think of descending a mountain.

```text
Very Small Steps
→ Safe but slow

Moderate Steps
→ Fast and controlled

Huge Steps
→ Keep jumping across the valley
```

The goal is not simply:

```text
Smaller Learning Rate
= Better
```

but:

```text
Appropriate Learning Rate
= Efficient Stable Learning
```

---

## 11. Batch Size

The **batch size** determines how many training examples are used to estimate a gradient before performing an update.

If:

```math
B=32
```

the model processes 32 examples before one optimizer step.

For a dataset containing:

```math
N
```

examples, the approximate number of updates per epoch is:

```math
\frac{N}{B}
```

---

## 12. Small Batch Size

Smaller batches produce noisier gradient estimates.

Conceptually:

```text
Small Batch
→ Less Data per Gradient
→ More Gradient Noise
→ More Frequent Updates
```

Potential advantages include:

- lower memory requirements
- more optimizer updates per epoch
- stochasticity that can sometimes aid generalization

Potential disadvantages include:

- noisy training
- less efficient hardware utilization
- unreliable BatchNorm statistics when extremely small

---

## 13. Large Batch Size

Larger batches use more examples per gradient estimate.

```text
Large Batch
→ More Data per Gradient
→ Smoother Gradient Estimate
→ Fewer Updates per Epoch
```

Potential advantages include:

- more stable gradients
- efficient parallel computation on GPUs
- higher computational throughput

Potential disadvantages include:

- greater memory requirements
- fewer parameter updates per epoch
- sometimes different generalization behaviour

---

## 14. Full-Batch Gradient Descent

At the extreme:

```math
B=N
```

the entire dataset is used for every gradient update.

This is **full-batch gradient descent**.

Conceptually:

```text
Entire Dataset
      ↓
One Gradient
      ↓
One Update
```

For large neural-network datasets this is often computationally impractical.

Mini-batch training therefore dominates modern deep learning.

---

## 15. Number of Epochs

An **epoch** means one complete pass through the training dataset.

If:

```text
Epochs = 20
```

the training algorithm processes the entire training dataset approximately 20 times.

More epochs allow more opportunities for learning.

But:

```text
More Epochs
≠ Automatically Better Model
```

Eventually the model may begin overfitting.

---

## 16. Epochs and Early Stopping

Suppose:

```text
Training Loss
→ Continues Decreasing

Validation Loss
→ Decreases
→ Reaches Minimum
→ Begins Increasing
```

Training beyond the validation minimum may increase overfitting.

Early stopping can therefore determine the effective number of epochs automatically.

Conceptually:

```text
Set Maximum Epochs
        +
Monitor Validation Performance
        ↓
Stop When Improvement Ends
```

---

## 17. Number of Hidden Layers

Network depth is itself a hyperparameter.

For example:

```text
Shallow Network

Input
 ↓
Hidden
 ↓
Output
```

versus:

```text
Deep Network

Input
 ↓
Hidden
 ↓
Hidden
 ↓
Hidden
 ↓
Hidden
 ↓
Output
```

Additional layers allow hierarchical representations to be learned.

However, unnecessary depth can increase:

- computation
- training difficulty
- parameter count
- overfitting risk

---

## 18. Number of Neurons

The width of hidden layers is also a hyperparameter.

For example:

```text
Dense(32)
```

versus:

```text
Dense(512)
```

A wider layer has greater representational capacity.

But:

```text
Too Few Neurons
→ Underfitting Risk

Too Many Neurons
→ More Computation
→ More Parameters
→ Greater Overfitting Potential
```

There is no universal ideal layer width.

---

## 19. Activation Function

Activation choice is also part of model design.

Examples include:

```text
ReLU
Leaky ReLU
Sigmoid
Tanh
Softmax
```

The correct activation depends partly on where it is used.

Typical hidden-layer choice:

```text
ReLU Family
```

Typical binary-classification output:

```text
Sigmoid
```

Typical multiclass output:

```text
Softmax
```

Activation functions influence both representational power and gradient behaviour.

---

## 20. Optimizer Choice

The optimizer is another hyperparameter.

Possible choices include:

```text
SGD
SGD + Momentum
RMSProp
Adam
AdamW
```

The optimizer controls how gradients are converted into parameter updates.

A useful starting point is often:

```text
General Experimentation
→ Adam / AdamW

Simple Baseline
→ SGD

Strong Tuned Training
→ SGD + Momentum can be excellent
```

The best optimizer remains problem-dependent.

---

## 21. Regularization Hyperparameters

Regularization introduces additional hyperparameters.

Examples include:

### 21.1. L2 / Weight Decay Strength

```math
\lambda
```

### 21.2. Dropout Rate

```math
p
```

For example:

```text
Dropout = 0.2
```

or:

```text
Dropout = 0.5
```

These values control how strongly the network is regularized.

---

## 22. Hyperparameter Tuning

**Hyperparameter tuning** is the process of searching for hyperparameter values that produce strong validation performance.

Conceptually:

```text
Choose Hyperparameters
        ↓
Train Model
        ↓
Evaluate Validation Set
        ↓
Try Different Hyperparameters
        ↓
Train Again
        ↓
Compare
        ↓
Select Best Configuration
```

The model parameters are learned **inside each training run**.

The hyperparameters are selected **across training runs**.

---

## 23. Training Set, Validation Set and Test Set

Hyperparameter tuning requires careful separation of data.

```text
Training Set
→ Learn Model Parameters

Validation Set
→ Choose Hyperparameters

Test Set
→ Final Unbiased Evaluation
```

This distinction is extremely important.

The test set should not repeatedly guide hyperparameter choices.

---

## 24. Why Not Tune on the Test Set?

Suppose we repeatedly try models and choose whichever performs best on the test set.

Then information from the test set influences model development.

Conceptually:

```text
Test Performance
      ↓
Influences Decisions
      ↓
Test Set Becomes Part of Development
```

The final test score is no longer an unbiased estimate of generalization.

Therefore:

```text
Training
→ Learn Parameters

Validation
→ Tune Hyperparameters

Test
→ Evaluate Once at the End
```

---

## 25. Manual Tuning

The simplest method is manual experimentation.

For example:

```text
Run 1
Learning Rate = 0.001
Batch Size = 32

Run 2
Learning Rate = 0.0001
Batch Size = 32

Run 3
Learning Rate = 0.001
Batch Size = 64
```

Results are compared using validation performance.

Manual tuning can work well when:

- the number of hyperparameters is small
- experience provides good starting values
- computational resources are limited

But it becomes inefficient for large search spaces.

---

## 26. Grid Search

**Grid search** defines a fixed set of candidate values.

Suppose:

```text
Learning Rate:
0.1, 0.01, 0.001

Batch Size:
32, 64, 128
```

Grid search evaluates every combination.

The number of experiments is:

```math
3\times3=9
```

If another hyperparameter has four choices:

```math
3\times3\times4=36
```

The search cost therefore grows rapidly.

---

## 27. Grid Search Example

Suppose:

```text
Learning Rate
→ [0.01, 0.001]

Batch Size
→ [32, 64]

Dropout
→ [0.2, 0.5]
```

The combinations are:

```math
2\times2\times2=8
```

training runs.

Grid search is systematic, but it can waste computation exploring combinations that matter little.

---

## 28. Random Search

**Random search** samples hyperparameter combinations from specified ranges or distributions.

Instead of testing every combination:

```text
Search Space
    ↓
Random Configuration
    ↓
Train
    ↓
Random Configuration
    ↓
Train
    ↓
...
```

This can be surprisingly effective.

---

## 29. Why Random Search Can Beat Grid Search

Suppose model performance depends strongly on learning rate but only weakly on another hyperparameter.

Grid search may repeatedly test the same few learning-rate values while varying the less important parameter.

Random search explores more distinct learning-rate values.

Conceptually:

```text
Grid Search
→ Uniform Coverage of Combinations

Random Search
→ Broader Coverage of Individual Values
```

In high-dimensional hyperparameter spaces, random search can therefore be more efficient.

---

## 30. Searching on a Logarithmic Scale

Some hyperparameters span several orders of magnitude.

Learning rate is a classic example.

Searching:

```text
0.00001
0.0001
0.001
0.01
0.1
```

is more meaningful than searching:

```text
0.01
0.02
0.03
0.04
0.05
```

because learning-rate behaviour often changes multiplicatively.

Therefore, learning rates and regularization strengths are frequently explored on a **logarithmic scale**.

---

## 31. Bayesian Optimization

More advanced tuning systems can use previous experiments to decide what to try next.

This family of techniques includes **Bayesian optimization**.

Conceptually:

```text
Try Configuration
       ↓
Observe Result
       ↓
Build Model of Search Space
       ↓
Choose Promising Next Configuration
       ↓
Repeat
```

Unlike grid or random search, later choices depend on earlier results.

This can reduce the number of expensive training runs.

---

## 32. Successive Halving

Another idea is to avoid spending equal resources on every candidate.

Suppose many configurations are tested briefly:

```text
20 Configurations
      ↓
Short Training
      ↓
Keep Best 10
      ↓
More Training
      ↓
Keep Best 5
      ↓
More Training
```

Poor configurations are terminated early.

This general strategy is known as **successive halving**.

It allocates more computational resources to promising candidates.

---

## 33. Hyperband

**Hyperband** builds upon resource-allocation ideas such as successive halving.

Instead of fully training every candidate, it explores many configurations while progressively allocating greater resources to promising ones.

The core idea is:

```text
Don't Spend Full Training Cost
on Obviously Poor Configurations
```

This can substantially reduce tuning cost.

---

## 34. Which Hyperparameters Matter Most?

Not every hyperparameter deserves equal attention.

A practical priority is often:

```text
1. Learning Rate

2. Architecture Size
   Layers / Units

3. Optimizer

4. Batch Size

5. Regularization
   Dropout / Weight Decay

6. Other Fine-Grained Choices
```

The exact ranking depends on the task.

But learning rate is almost always among the most influential.

---

## 35. Tune Coarsely Before Tuning Finely

Suppose the useful learning-rate region is unknown.

It makes little sense to begin comparing:

```text
0.0010
vs
0.0011
```

Instead, first test:

```text
0.1
0.01
0.001
0.0001
```

Once a promising region is found, refine it.

Conceptually:

```text
Broad Search
    ↓
Find Useful Region
    ↓
Narrow Search
    ↓
Fine Tuning
```

This saves computation.

---

## 36. Hyperparameters Interact

Hyperparameters should not always be considered independently.

For example:

```text
Learning Rate
↔ Batch Size
```

and:

```text
Network Size
↔ Regularization Strength
```

A larger model may require stronger regularization.

A different batch size may change the most effective learning rate.

Therefore:

```text
Best Hyperparameter A
when B = x

may not be

Best Hyperparameter A
when B = y
```

Hyperparameter tuning searches configurations, not isolated numbers.

---

## 37. Validation Performance Can Be Noisy

Neural-network training contains randomness from:

- weight initialization
- mini-batch ordering
- dropout
- stochastic optimization

Therefore, two runs using identical hyperparameters can produce somewhat different results.

Conceptually:

```text
Same Hyperparameters
+
Different Randomness
→ Slightly Different Models
```

For important comparisons, multiple runs with different random seeds may provide a more reliable estimate.

---

## 38. Reproducibility During Tuning

Random seeds can help make comparisons fairer.

For example:

```text
Configuration A
→ Seed 42

Configuration B
→ Seed 42
```

This reduces one source of variation.

However, a configuration that performs well only under one lucky seed may not be robust.

For final evaluation, performance across multiple seeds can be useful.

---

## 39. Computational Cost

Hyperparameter tuning can be expensive.

If one training run takes:

```math
2\text{ hours}
```

and grid search requires:

```math
50
```

runs, the total raw compute is:

```math
100\text{ hours}
```

before considering parallel execution.

This is why efficient search strategies and early termination methods are important in deep learning.

---

## 40. Tuning Is an Outer Optimization Loop

There are effectively two optimization processes.

### 40.1. Inner Loop

The neural network learns parameters:

```text
Weights
Biases
```

using:

```text
Backpropagation
+
Optimizer
```

### 40.2. Outer Loop

We search for hyperparameters:

```text
Learning Rate
Architecture
Batch Size
Regularization
...
```

using validation performance.

Conceptually:

```text
Outer Loop:
Choose Hyperparameters
        ↓
    Inner Loop:
    Train Parameters
        ↓
Validation Performance
        ↓
Choose Better Hyperparameters
```

This distinction provides a powerful mental model for hyperparameter tuning.

---

## 41. Hyperparameter Tuning and Overfitting

It is possible to overfit the validation set.

Suppose hundreds or thousands of configurations are repeatedly compared on the same validation data.

Eventually, development decisions may become overly tailored to that validation set.

Therefore:

```text
Training Set
→ Parameter Learning

Validation Set
→ Development Decisions

Test Set
→ Final Independent Check
```

For large projects, additional validation strategies or cross-validation may be used.

---

## 42. Cross-Validation in Deep Learning

In classical machine learning, `k`-fold cross-validation is common.

The dataset is divided into `k` folds, and models are repeatedly trained using different validation folds.

However, deep neural networks can be expensive to train.

Therefore full cross-validation may require substantial computation.

Deep-learning workflows often use:

```text
Train Split
+
Validation Split
+
Test Split
```

especially when datasets are large.

Cross-validation remains useful when data is limited and computational cost is manageable.

---

## 43. Keep an Experiment Record

Hyperparameter tuning quickly becomes confusing without systematic tracking.

For each run, record:

```text
Architecture
Learning Rate
Optimizer
Batch Size
Dropout
Weight Decay
Epochs
Random Seed
Training Metrics
Validation Metrics
```

Otherwise:

```text
"I think that 0.001 run was better..."
```

quickly becomes unreliable.

Experiment tracking is therefore an important engineering practice, not mere administrative work.

---

## 44. Do Not Tune Everything at Once

A practical workflow is:

```text
Build Reasonable Baseline
        ↓
Verify Training Works
        ↓
Tune High-Impact Hyperparameters
        ↓
Address Underfitting / Overfitting
        ↓
Fine-Tune Remaining Choices
```

Searching an enormous hyperparameter space before confirming that the pipeline itself works wastes computation.

Always establish a functioning baseline first.

---

## 45. Diagnose Before Tuning

Hyperparameter tuning should be guided by observed behaviour.

### 45.1. If Training Loss Is High

Possible actions:

```text
Increase Model Capacity
Train Longer
Check Learning Rate
Reduce Excessive Regularization
```

### 45.2. If Training Loss Is Low but Validation Loss Is High

Possible actions:

```text
Increase Regularization
Add Dropout
Use Weight Decay
Use Data Augmentation
Reduce Model Capacity
Get More Data
```

### 45.3. If Training Is Unstable

Possible actions:

```text
Lower Learning Rate
Check Initialization
Use Normalization
Check Gradient Magnitudes
Use Gradient Clipping
```

Tuning becomes far more efficient when driven by diagnosis.

---

## 46. Hyperparameter Tuning Workflow

A practical sequence is:

```text
1. Build Baseline Model
        ↓
2. Confirm Pipeline Works
        ↓
3. Choose Validation Metric
        ↓
4. Tune Learning Rate
        ↓
5. Tune Architecture Capacity
        ↓
6. Tune Regularization
        ↓
7. Tune Batch Size / Optimizer
        ↓
8. Compare Best Configurations
        ↓
9. Retrain Final Configuration
        ↓
10. Evaluate on Test Set
```

The exact order can vary, but this gives a disciplined starting point.

---

## 47. Hyperparameters in the Complete Learning System

We can now separate the entire training system into two categories.

### 47.1. Learned During Training

```text
Weights
Biases
BatchNorm γ and β
Other Trainable Parameters
```

### 47.2. Chosen Around Training

```text
Architecture
Learning Rate
Optimizer
Batch Size
Epochs
Dropout Rate
Weight Decay
Initialization
Learning Rate Schedule
```

Training then becomes:

```text
Choose Hyperparameters
        ↓
Initialize Model
        ↓
Forward Propagation
        ↓
Loss
        ↓
Backpropagation
        ↓
Optimizer
        ↓
Learn Parameters
        ↓
Validation
        ↓
Adjust Hyperparameters
        ↓
Train Again
```

---

## 48. Key Takeaways

- Hyperparameters control model architecture or training behaviour.
- Model parameters are learned through backpropagation and optimization.
- Optimizer state is different from both parameters and hyperparameters.
- Learning rate is one of the most important hyperparameters.
- A learning rate that is too small produces slow training.
- A learning rate that is too large can cause oscillation or divergence.
- Batch size controls how many examples contribute to each gradient estimate.
- Small batches produce noisier gradients and more frequent updates.
- Large batches produce smoother gradient estimates and require more memory.
- The number of epochs determines how many times the model processes the training dataset.
- Network depth and width determine model capacity.
- Optimizer choice is itself a hyperparameter.
- Dropout rate and weight decay are regularization hyperparameters.
- Hyperparameter tuning uses validation performance to compare configurations.
- The test set should not be repeatedly used for tuning.
- Grid search evaluates predefined combinations systematically.
- Random search samples combinations and can be more efficient in large search spaces.
- Learning rates and regularization strengths are often searched logarithmically.
- Bayesian optimization uses previous experiments to guide future trials.
- Successive halving and Hyperband terminate weak candidates early.
- Hyperparameters interact with one another.
- Neural-network results can vary because training is stochastic.
- Experiment tracking is essential when comparing many configurations.
- Diagnose the model's problem before blindly changing hyperparameters.
- Hyperparameter tuning forms an outer optimization loop around parameter learning.

### 48.1. Memory Hook

```text
Parameters
→ Learned

Hyperparameters
→ Chosen

Optimizer State
→ Remembered During Training


High-Impact Hyperparameters:

Learning Rate
→ How Big a Step?

Batch Size
→ How Much Data per Step?

Epochs
→ How Long to Train?

Layers / Neurons
→ How Much Capacity?

Dropout / Weight Decay
→ How Much Regularization?

Optimizer
→ How to Use Gradients?


Tuning:

Training Set
→ Learn Parameters

Validation Set
→ Choose Hyperparameters

Test Set
→ Final Evaluation


Search Methods:

Manual
→ Human Choice

Grid Search
→ Try Every Combination

Random Search
→ Sample Combinations

Bayesian Optimization
→ Learn Where to Search

Hyperband
→ Stop Bad Runs Early


Core Idea:

Parameters are what the network learns.

Hyperparameters determine
how we let it learn.
```
