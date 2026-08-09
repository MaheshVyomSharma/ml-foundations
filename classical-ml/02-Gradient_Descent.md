# 02. Gradient Descent

> *"Great achievements are made by a series of small steps."*

---

# Why Should I Care?

Almost every Machine Learning model learns by improving itself little by little.

But how does a model know whether it is improving?

How does it know which parameters to change?

Gradient Descent answers these questions.

It is one of the most widely used optimization algorithms in Machine Learning and forms the foundation of many advanced algorithms, including Logistic Regression and Neural Networks.

---

# Five-Minute Story

Imagine standing halfway down a mountain on a foggy day.

You cannot see the valley below.

You cannot see the peak above.

The only information available is the slope beneath your feet.

If the ground slopes downward to your left, you step left.

If it slopes downward behind you, you step back.

After taking a small step, you check the slope again.

By repeating this process, you eventually reach the bottom of the valley.

Gradient Descent works in exactly the same way.

Instead of walking down a mountain, it gradually adjusts a model's parameters until the prediction error becomes as small as possible.

---

# Learning Objectives

After reading this chapter, you should be able to:

- Explain the purpose of Gradient Descent.
- Understand the concepts of cost function and gradient.
- Describe how Gradient Descent updates model parameters.
- Explain the role of the learning rate.
- Distinguish between Batch, Stochastic, and Mini-batch Gradient Descent.

---

# What is Gradient Descent?

Gradient Descent is an **optimization algorithm** used to minimize the prediction error of a Machine Learning model by repeatedly updating its parameters in the direction that reduces the error.

Simply put,

> Gradient Descent helps a model learn by making small improvements over many iterations.

---

# Problem It Solves

Suppose a Linear Regression model predicts house prices poorly.

The model needs to answer two questions:

1. How wrong am I?
2. What should I change to become less wrong?

The **cost function** answers the first question.

The **gradient** answers the second.

Gradient Descent repeatedly uses both to improve the model.

---

# Key Terminology

| Term | Meaning |
|------|---------|
| Parameter | A value learned by the model (for example, slope or intercept). |
| Cost Function | A numerical measure of prediction error. Lower is better. |
| Gradient | The direction of the steepest increase in the cost function. |
| Iteration | One complete parameter update. |
| Learning Rate | The size of each update step. |

---

# Intuition

Imagine rolling a ball down a hill.

The ball naturally moves toward lower ground.

Similarly, Gradient Descent moves the model toward lower prediction error.

Unlike a real ball, however, the algorithm does **not** roll automatically.

It calculates the direction to move and deliberately takes a small step.

This process repeats until further improvement becomes negligible.

---

# Mental Picture

Imagine wearing a blindfold while walking down a hill.

You cannot see where the lowest point is.

Instead, you use the slope beneath your feet.

Take one careful step.

Stop.

Feel the slope again.

Repeat.

Eventually, you arrive near the bottom.

That is Gradient Descent.

---

# How Gradient Descent Works

The learning process follows these steps:

1. Start with random parameter values.
2. Make predictions.
3. Measure prediction error using a cost function.
4. Compute the gradient.
5. Update the parameters.
6. Repeat until the cost stops decreasing significantly.

```text
Initial Parameters
        │
        ▼
Make Predictions
        │
        ▼
Calculate Cost
        │
        ▼
Calculate Gradient
        │
        ▼
Update Parameters
        │
        ▼
Repeat
```

---

# Why Move in the Negative Direction?

The gradient always points toward the direction of **maximum increase** in the cost function.

Our objective is the opposite—we want to reduce the cost.

Therefore, the parameters are updated in the **negative direction of the gradient**, causing the model to move "downhill" toward lower error.

---

# Learning Rate

The **learning rate** determines the size of each update.

### If the learning rate is too small

- Learning becomes very slow.
- Many iterations are required.

### If the learning rate is too large

- The algorithm may overshoot the optimum.
- Learning may become unstable.
- The model may fail to converge.

Choosing an appropriate learning rate is therefore important for efficient learning.

---

# Types of Gradient Descent

## Batch Gradient Descent

Uses the entire training dataset to calculate each update.

**Advantages**

- Stable updates.
- Smooth convergence.

**Limitations**

- Slow on very large datasets.

---

## Stochastic Gradient Descent (SGD)

Updates the parameters after processing **one training sample**.

**Advantages**

- Faster updates.
- Suitable for large datasets.

**Limitations**

- More fluctuation during learning.

---

## Mini-batch Gradient Descent

Updates parameters using a small group of samples.

This combines the advantages of Batch Gradient Descent and SGD.

It is the most commonly used approach in modern Machine Learning.

---

# Advantages

- Simple and effective.
- Scales to large datasets.
- Applicable to many Machine Learning algorithms.
- Forms the basis of deep learning optimization.

---

# Limitations

- Choice of learning rate is important.
- Can converge slowly.
- May become trapped in local minima or saddle points for complex optimization problems.
- Requires multiple iterations before convergence.

---

# Common Applications

Gradient Descent is used in:

- Linear Regression
- Logistic Regression
- Neural Networks
- Deep Learning
- Many optimization problems beyond Machine Learning

---

# Memory Hook

> **Gradient tells you where the hill climbs fastest.**
>
> **Gradient Descent walks in the opposite direction.**

---

# Common Mistakes

- Thinking the gradient always points downhill.
- Choosing an excessively large learning rate.
- Assuming Gradient Descent always reaches the global minimum.
- Confusing the cost function with the gradient.

---

# Frequently Asked Questions

### Is Gradient Descent a Machine Learning model?

No.

It is an optimization algorithm used to train many Machine Learning models.

---

### Does every Machine Learning algorithm use Gradient Descent?

No.

Some algorithms, such as Decision Trees and Random Forests, use entirely different learning strategies.

---

### Why does learning take many iterations?

Each update is intentionally small to allow the model to improve gradually without making unstable jumps.

---

# 30-Second Revision

- Optimization algorithm.
- Minimizes prediction error.
- Uses the gradient to determine update direction.
- Moves opposite to the gradient.
- Learning rate controls step size.
- Types: Batch, Stochastic, Mini-batch.
- Used extensively in Machine Learning and Deep Learning.

---

# Looking Ahead

Linear Regression predicts continuous values.

Gradient Descent explains **how** many Machine Learning models learn.

The next chapter introduces **Logistic Regression**, which uses the same optimization idea to solve **classification** problems instead of regression.