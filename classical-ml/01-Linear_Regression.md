# 01. Linear Regression

> *"All models are wrong, but some are useful."*  
> — **George E. P. Box**

---

# Why Should I Care?

Many real-world problems involve predicting a **continuous numerical value**.

Examples include:

- Estimating the price of a house
- Predicting tomorrow's temperature
- Forecasting monthly sales
- Estimating a person's salary based on experience

Linear Regression is one of the simplest and most widely used algorithms for solving such problems. It also forms the foundation for understanding several advanced Machine Learning techniques.

---

# Five-Minute Story

A real estate company has records of thousands of houses.

For every house, it knows:

- Floor area
- Number of bedrooms
- Age of the property
- Distance from the city centre

It also knows the final selling price.

Now a new house comes onto the market.

Its selling price is unknown.

Instead of asking an expert to estimate the price manually, can a computer learn from the previous houses and predict a reasonable selling price?

This is exactly the type of problem that **Linear Regression** is designed to solve.

---

# Learning Objectives

After reading this chapter, you should be able to:

- Explain what Linear Regression is.
- Identify problems suitable for Linear Regression.
- Understand how the algorithm learns from data.
- Interpret predictions made by the model.
- Evaluate a Linear Regression model using common metrics.

---

# What is Linear Regression?

**Linear Regression** is a **supervised learning algorithm** used to model the relationship between one or more **features** and a **continuous target variable** by fitting the best possible linear relationship between them.

In simple terms,

> Linear Regression learns how the target changes as one or more input features change.

---

# Problem It Solves

Linear Regression is used when:

- The target variable is **continuous**.
- Historical labelled data is available.
- The relationship between features and the target is approximately linear.

### Suitable Problems

- House price prediction
- Stock demand forecasting
- Sales prediction
- Electricity consumption estimation
- Insurance premium estimation

### Not Suitable For

- Spam detection
- Image classification
- Customer segmentation
- Face recognition

These require different types of Machine Learning models.

---

# Intuition

Imagine plotting house prices against their floor areas.

The data points will usually not lie perfectly on a straight line.

Linear Regression attempts to draw **the line that best represents the overall trend** in the data.

Some houses will lie above the line.

Some will lie below it.

The goal is **not** to pass through every point, but to find the line that represents the data as accurately as possible.

---

# Mental Picture

Imagine placing a transparent ruler over a scatter plot.

You rotate and slide the ruler until the overall distance between the ruler and all the data points becomes as small as possible.

That ruler represents the regression line.

---

# Mathematical Foundation

For a single feature, Linear Regression models the relationship as:

**Prediction = Intercept + (Slope × Feature)**

where:

- **Intercept** is the predicted value when the feature equals zero.
- **Slope** represents how much the prediction changes for every one-unit increase in the feature.

For multiple features, the same idea extends by adding the contribution of each feature to the final prediction.

> **Note:** You do not need to memorise the equation to understand how Linear Regression works. The important idea is that the prediction is a weighted combination of the input features.

---

# How the Model Learns

During training, the algorithm:

1. Makes an initial prediction.
2. Compares the prediction with the actual target.
3. Measures the prediction error.
4. Adjusts the model parameters to reduce the error.
5. Repeats the process until the model finds the best-fitting relationship.

This process is called **training** or **model fitting**.

---

# Workflow

```text
Training Data
      │
      ▼
Learn Relationship
      │
      ▼
Fit Regression Line
      │
      ▼
Predict New Values
```

---

# Advantages

- Simple to understand and interpret.
- Fast to train.
- Computationally efficient.
- Performs well when relationships are approximately linear.
- Serves as a strong baseline model for many regression problems.

---

# Limitations

- Assumes an approximately linear relationship.
- Sensitive to outliers.
- Cannot model complex non-linear patterns without feature engineering.
- Performance decreases when assumptions are violated.

---

# Common Evaluation Metrics

Regression models are commonly evaluated using:

| Metric | Purpose |
|---------|---------|
| MAE | Average prediction error |
| MSE | Penalises larger errors more heavily |
| RMSE | Error expressed in the original unit of the target |
| R² Score | Measures how well the model explains the variation in the target |

These metrics are discussed in detail in the **Model Evaluation** chapter.

---

# Common Applications

Linear Regression is widely used in:

- Real estate price prediction
- Financial forecasting
- Sales forecasting
- Demand estimation
- Risk analysis
- Manufacturing analytics

---

# Memory Hook

> **Regression predicts numbers.**
>
> Think of Linear Regression as finding the **best possible straight ruler** through a cloud of data points.

---

# Common Mistakes

- Using Linear Regression for classification problems.
- Assuming a perfect fit is always desirable.
- Ignoring outliers.
- Interpreting correlation as causation.
- Forgetting that prediction accuracy depends on data quality.

---

# Frequently Asked Questions

### Is Linear Regression supervised or unsupervised?

Supervised, because the correct target values are known during training.

---

### Can Linear Regression handle multiple input features?

Yes.

Using one feature is called **Simple Linear Regression**.

Using two or more features is called **Multiple Linear Regression**.

---

### Does Linear Regression always produce a straight line?

For one feature, yes.

For multiple features, the model learns a higher-dimensional linear relationship, commonly called a **hyperplane**.

---

# 30-Second Revision

- Supervised learning algorithm.
- Predicts continuous numerical values.
- Learns a linear relationship between features and target.
- Fast, interpretable, and widely used.
- Works well for approximately linear data.
- Common metrics: MAE, MSE, RMSE, R².
- Foundation for many advanced Machine Learning algorithms.

---

# Looking Ahead

Linear Regression predicts **numbers**.

But many Machine Learning problems do not ask for a numerical value.

Instead, they ask questions such as:

- Will the customer buy the product?
- Is this email spam?
- Does the patient have the disease?

These are **classification problems**, where the answer belongs to a category rather than a continuous value.

In the next chapter, we explore **Logistic Regression**, which extends the ideas of Linear Regression to solve classification problems.