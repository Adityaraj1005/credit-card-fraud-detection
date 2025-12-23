# Credit Card Fraud Detection

## Project Topic
Credit Card Fraud Detection using Machine Learning

## Why Credit Card Fraud Detection?

Credit card fraud detection is a real-world classification problem where the goal is to identify fraudulent transactions among millions of legitimate ones.

This problem is important because:
- Fraudulent transactions cause huge financial losses to banks and customers.
- The dataset is highly imbalanced, which makes it a challenging and realistic machine learning problem.
- It helps demonstrate how different ML models perform on imbalanced data using metrics like Precision, Recall, and F1-score instead of just Accuracy.
- This project reflects how machine learning is applied in the finance and banking industry for risk management.

I chose this topic to gain hands-on experience with:
- Binary classification
- Imbalanced datasets
- Model evaluation using confusion matrix, precision, recall, and F1-score
- Comparing multiple machine learning algorithms


## Dataset Source

The dataset used in this project is the **Credit Card Fraud Detection dataset** available on **Kaggle**.

Source:
```
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

This dataset contains real credit card transactions made by European cardholders over a period of two days.


## Dataset Description

The dataset consists of **284,807 transactions**, out of which **492 are fraudulent**.
This makes the dataset **highly imbalanced**, where fraud cases represent only about **0.17%** of the total data.

### Features in the Dataset

- **Time**  
  Represents the time elapsed in seconds since the first transaction in the dataset.

- **V1 to V28**  
  These are anonymized features obtained using **Principal Component Analysis (PCA)**.  
  The original feature names and values are hidden to protect sensitive financial information.

- **Amount**  
  The transaction amount. This feature is important because fraud transactions often involve unusual amounts.

- **Class (Target Variable)**  
  - `0` → Legitimate transaction  
  - `1` → Fraudulent transaction


### Why are V1–V28 PCA Transformed?

To maintain confidentiality, the original transaction details (like merchant, location, etc.) were transformed using **PCA**.
This ensures:
- User privacy is preserved
- Feature correlation is reduced
- Models perform better and train faster


## Data Preprocessing

Before training the machine learning models, several preprocessing steps were applied to prepare the data for optimal performance.


### Handling Class Imbalance

The dataset is highly imbalanced, with fraudulent transactions making up less than 1% of the data.
To ensure that both classes are fairly represented during training and testing, **Stratified Train-Test Split** was used.

## Missing Data Check

The dataset was checked for missing (null) values using:

```python
dataset.isnull().sum()
```

## Why the Dataset Contains No Missing Values

This dataset contains no missing values **by design**.

**Reasons:**
- Features **V1–V28** were created using **PCA**, which requires a complete numerical dataset
- Incomplete records cannot be used in PCA and are removed during preprocessing
- Each row represents a **completed credit card transaction**, where time and amount are always recorded
- The dataset was **preprocessed and validated** for research use before release

**Conclusion:**
Since the data was already complete and PCA-transformed, **no missing data handling was required**.


### Train-Test Split

The dataset was split into training and testing sets using **stratified sampling**.

Stratification ensures that:
- The proportion of fraud and non-fraud transactions remains the same in both training and test sets
- The model does not become biased toward the majority class

This is especially important for imbalanced datasets like fraud detection.


### Feature Scaling

The **Time** and **Amount** features were scaled using **StandardScaler**.

Reason:
- PCA features (V1–V28) are already on a similar scale
- Time and Amount have larger value ranges and can bias the model
- Scaling improves the performance of **Logistic Regression**


### Why StandardScaler?

StandardScaler scales features so that:
- Mean becomes 0
- Standard deviation becomes 1

This prevents features with large values from dominating the model and helps **Logistic Regression** work better.


## Models Used and Results

To solve the credit card fraud detection problem, multiple machine learning models were trained and evaluated to handle highly imbalanced data.

### 1. Logistic Regression
Logistic Regression was used as a baseline model for binary classification.

**Why used:**
- Simple and fast baseline model
- Works well with scaled numerical features
- Easy to interpret

**Results:**
- Accuracy: 0.99917  
- Precision: 0.849  
- Recall: 0.633  
- F1-score: 0.725  

---

### 2. Random Forest
Random Forest is an ensemble learning method that combines multiple decision trees to improve performance and reduce overfitting.

**Why used:**
- Handles non-linear relationships
- More robust than a single model
- Performs well on imbalanced datasets

**Results:**
- Accuracy: 0.99944  
- Precision: 0.958  
- Recall: 0.704  
- F1-score: 0.812  

---

### 3. XGBoost
XGBoost (Extreme Gradient Boosting) is a powerful boosting algorithm that builds models sequentially to correct previous errors.

**Why used:**
- Strong performance on imbalanced datasets
- Handles complex patterns efficiently
- Widely used in real-world fraud detection problems

**Results:**
- Accuracy: 0.99940  
- Precision: 0.890  
- Recall: 0.745  
- F1-score: 0.811  

---

### Model Comparison Summary
Among all models, **Random Forest** achieved the best overall performance in terms of F1-score, while **XGBoost** showed better recall, making both suitable for fraud detection.


## Model Evaluation and Results

The models were evaluated using a **confusion matrix** and performance metrics like **Accuracy, Precision, Recall, and F1-score**.

### Confusion Matrix

A confusion matrix shows the counts of:
- **True Positives (TP):** Fraudulent transactions correctly detected as fraud  
- **True Negatives (TN):** Legitimate transactions correctly detected as legitimate  
- **False Positives (FP):** Legitimate transactions incorrectly detected as fraud  
- **False Negatives (FN):** Fraudulent transactions missed by the model  


### Metrics

1. **Accuracy**
- Proportion of correct predictions (TP + TN) out of all predictions  
- **Limitation:** Can be misleading for imbalanced data like fraud detection  
  - Here, fraud is <1% of data, so high accuracy alone is not enough

2. **Precision**
- Proportion of predicted fraud transactions that are actually fraud  
- **Formula:** TP / (TP + FP)  
- **Importance:** Shows how many flagged fraud cases are correct

3. **Recall (Sensitivity)**
- Proportion of actual fraud transactions that were correctly detected  
- **Formula:** TP / (TP + FN)  
- **Importance:** Shows how well the model identifies real fraud cases  
- **Critical in fraud detection** because missing fraud (FN) is costly

4. **F1-Score**
- Harmonic mean of precision and recall  
- **Formula:** 2 × (Precision × Recall) / (Precision + Recall)  
- Balances both **false positives** and **false negatives**  
- **Important here** because we want to detect fraud while avoiding too many false alarms


## Model Performance Comparison

| Model               | Accuracy   | Precision | Recall  | F1-Score |
|--------------------|-----------|----------|--------|----------|
| Logistic Regression | 0.99917   | 0.849    | 0.633  | 0.725    |
| Random Forest       | 0.99944   | 0.958    | 0.704  | 0.812    |
| XGBoost             | 0.99940   | 0.890    | 0.745  | 0.811    |

**Notes:**
- Accuracy is high for all models due to the imbalanced dataset
- **Recall** is especially important in fraud detection to catch as many frauds as possible
- **F1-score** balances precision and recall, giving a better overall performance measure

---

## Conclusion

- The dataset was clean and complete, requiring **no missing data handling**  
- Features were scaled appropriately, with PCA features already normalized  
- Three models were trained: Logistic Regression, Random Forest, and XGBoost  
- **Random Forest** achieved the highest F1-score, while **XGBoost** showed slightly higher recall  
- This project demonstrates how machine learning can effectively detect fraud in imbalanced datasets  
- Using metrics like **precision, recall, and F1-score** is more meaningful than accuracy alone for fraud detection
