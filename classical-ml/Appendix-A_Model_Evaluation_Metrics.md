# Appendix A. Model Evaluation Metrics

This appendix collects the formulas behind the metrics introduced in Chapter 12. The formulas assume a dataset of $n$ observations and predictions $\hat{y}_i$ for observed values $y_i$, unless stated otherwise.

## 1. Classification Notation

For binary classification, designate one class as **Positive** and the other as **Negative**.

| Symbol | Meaning |
| -------- | --------- |
| TP | True Positive: predicted Positive and actually Positive |
| TN | True Negative: predicted Negative and actually Negative |
| FP | False Positive: predicted Positive but actually Negative |
| FN | False Negative: predicted Negative but actually Positive |
| $P$ | Number of actual Positives, $P = TP + FN$ |
| $N$ | Number of actual Negatives, $N = TN + FP$ |

The total number of observations is $TP + TN + FP + FN$.

## 2. Classification Metrics

### 2.1. Accuracy

Accuracy is the proportion of all predictions that are correct.

```math
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

### 2.2. Error Rate

Error rate is the proportion of all predictions that are incorrect.

```math
\text{Error Rate} = \frac{FP + FN}{TP + TN + FP + FN} = 1 - \text{Accuracy}
```

### 2.3. Precision

Precision is the proportion of predicted Positives that are truly Positive.

```math
\text{Precision} = \frac{TP}{TP + FP}
```

### 2.4. Recall or Sensitivity

Recall is the proportion of actual Positives that the model identifies.

```math
\text{Recall} = \text{Sensitivity} = \text{TPR} = \frac{TP}{TP + FN}
```

### 2.5. Specificity

Specificity is the proportion of actual Negatives that the model identifies.

```math
\text{Specificity} = \text{TNR} = \frac{TN}{TN + FP}
```

The false-positive rate is the complement of specificity:

```math
\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}
```

### 2.6. F1 Score

The F1 score is the harmonic mean of Precision and Recall. It is high only when both are high.

```math
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

Equivalently, in terms of the confusion-matrix counts:

```math
F_1 = \frac{2TP}{2TP + FP + FN}
```

### 2.7. Balanced Accuracy

Balanced accuracy gives equal weight to each class, which is useful when class sizes differ substantially.

```math
\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}
```

## 3. Threshold-Based Metrics

Many classifiers produce a score or probability. A threshold converts that value into a class prediction. Changing the threshold changes TP, TN, FP, and FN, and therefore changes the metrics.

### 3.1. ROC Curve and ROC-AUC

The ROC curve plots the True Positive Rate against the False Positive Rate as the classification threshold varies:

```math
\text{TPR} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{FP + TN}
```

ROC-AUC is the area under this curve:

```math
\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR})\,d(\text{FPR})
```

It can also be interpreted as the probability that a randomly chosen Positive receives a higher model score than a randomly chosen Negative.

### 3.2. Precision-Recall Curve and Average Precision

The precision-recall curve plots Precision against Recall as the threshold varies. Average Precision summarizes this curve by weighting precision by the increase in recall:

```math
\text{AP} = \sum_k (R_k - R_{k-1})P_k
```

where $P_k$ and $R_k$ are the precision and recall at threshold $k$. This curve is often more informative than ROC-AUC when the Positive class is rare.

## 4. Regression Metrics

For regression, define the residual for observation $i$ as:

```math
e_i = y_i - \hat{y}_i
```

### 4.1. Mean Absolute Error (MAE)

MAE is the average absolute difference between observed and predicted values.

```math
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert
```

### 4.2. Mean Squared Error (MSE)

MSE is the average squared residual. Squaring gives larger errors more influence.

```math
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
```

### 4.3. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE, so it has the same units as the target variable.

```math
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
```

### 4.4. Coefficient of Determination ($R^2$)

Let $\bar{y}$ be the mean of the observed target values. $R^2$ compares the model's residual sum of squares with the total sum of squares.

```math
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
```

The baseline in this comparison predicts $\bar{y}$ for every observation. $R^2=1$ indicates perfect predictions; $R^2=0$ means the model is no better than that baseline on this measure; and $R^2$ can be negative when it performs worse than the baseline.

### 4.5. Mean Absolute Percentage Error (MAPE)

MAPE expresses absolute errors as percentages of the observed values.

```math
\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left\lvert\frac{y_i - \hat{y}_i}{y_i}\right\rvert
```

MAPE is undefined when $y_i=0$ and can become disproportionately large when observed values are close to zero.

## 5. Choosing a Metric

No metric is universally best. Choose one whose error meaning matches the application:

| Situation | Useful metric |
| --------- | --------------- |
| Classes are balanced and errors have similar cost | Accuracy |
| False Positives are costly | Precision |
| False Negatives are costly | Recall or Sensitivity |
| Both Precision and Recall matter | F1 score |
| Classes are imbalanced | Balanced accuracy, F1 score, or a precision-recall curve |
| Regression errors should be easy to interpret | MAE |
| Large regression errors should be penalized strongly | MSE or RMSE |
| Relative error is meaningful and targets are nonzero | MAPE |
