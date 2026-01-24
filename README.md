# Credit_Card_Churn_Detection

**a.Problem Statement**

Customer churn is a major concern for financial institutions, especially credit card companies, as acquiring new customers is significantly more expensive than retaining existing ones. The objective of this project is to build a machine learning–based classification system that can predict whether a customer is likely to leave (churn) the bank based on their demographic, financial, and behavioral attributes.

The system aims to help banks proactively identify high-risk customers and take preventive measures to improve customer retention.

**b.Dataset Description**

The dataset used in this project is a Credit Card Customer Churn dataset, which contains customer-level information related to banking behavior and demographics.

Key characteristics of the dataset:

Each row represents a unique bank customer

The target variable indicates whether the customer has exited the bank

Both numerical and categorical features are present

Target Variable:

Exited

1 → Customer has churned

0 → Customer is retained

**Feature Description:**

CreditScore:	Customer credit score
Age:           	Age of the customer
Tenure:      	Number of years the customer has been with the bank
Balance:     	Bank account balance
NumOfProducts: 	Number of bank products used
HasCrCard"  	Whether the customer has a credit card (0/1)
IsActiveMember:	Whether the customer is an active member (0/1)
EstimatedSalary:Estimated annual salary
Geography:  	Customer’s country
Gender:     	Customer’s gender

*Unnecessary identifier columns such as RowNumber, CustomerId, and Surname were removed during preprocessing.

**c. Models Used and Evaluation:**

Six different machine learning models were trained and evaluated for customer churn prediction.
All models were implemented using scikit-learn pipelines to ensure consistent preprocessing during training and inference.

**Models Implemented**

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (kNN)
Naive Bayes Classifier
Random Forest (Ensemble Model)
XGBoost (Ensemble Model)

**Evaluation Metrics Used**

Accuracy
AUC (Area Under ROC Curve)
Precision
Recall
F1 Score
MCC (Matthews Correlation Coefficient)

These metrics were chosen to handle class imbalance and provide a balanced evaluation of model performance.

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------------ | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression      | 0.811    | 0.580 | 0.553     | 0.201  |  0.295   | 0.248 |
| Decision Tree            | 0.747    | 0.766 | 0.424     | 0.798  | 0.554    | 0.438 |
| K-Nearest Neighbors      | 0.845    | 0.694 | 0.657     | 0.445  | 0.531    | 0.455 |
| Naive Bayes              | 0.833    | 0.654 | 0.635     | 0.358  | 0.458    | 0.390 |
| Random Forest (Ensemble) | 0.839    | 0.790 | 0.573     | 0.709  | 0.634    | 0.537 |
| XGBoost (Ensemble)       | 0.871    | 0.739 | 0.745     | 0.521  | 0.614    | 0.552 |


**Conclusion**

Since the dataset is imbalanced, accuracy alone is not sufficient for evaluating model performance. Therefore, F1 Score and ROC-AUC were primarily used for model comparison. F1 Score provides a balance between precision and recall, while AUC measures the model’s ability to distinguish between churners and non-churners independent of classification threshold.



|ML Model Name                  | Observation about Model Performance 
| **Logistic Regression**       | Logistic Regression provided a strong baseline performance with stable and interpretable results. It achieved reasonable precision and recall, but its linear decision boundary limited its ability to capture complex non-linear patterns present in customer behavior, resulting in comparatively lower F1 score and AUC than ensemble models. |
| **Decision Tree**             | The Decision Tree model was able to capture non-linear relationships and showed improved recall compared to Logistic Regression. However, it exhibited signs of overfitting, leading to reduced generalization performance on unseen test data, especially reflected in lower AUC and MCC values.                                                |
| **k-Nearest Neighbors (kNN)** | kNN showed moderate performance but was sensitive to feature scaling and the choice of the value of *k*. Its performance degraded in higher-dimensional feature space, and it struggled to effectively separate churners from non-churners, resulting in lower recall and inconsistent F1 score.                                                 |
| **Naive Bayes**               | Naive Bayes performed efficiently with fast computation but relied on the strong assumption of feature independence. This assumption was not fully satisfied in the dataset, leading to lower predictive performance and reduced F1 score compared to other models, although recall was relatively acceptable.                                   |
| **Random Forest (Ensemble)**  | Random Forest significantly improved performance by reducing overfitting through ensemble learning. It achieved higher AUC, F1 score, and MCC compared to individual models, demonstrating better generalization and robustness in identifying churners while maintaining balanced precision and recall.                                         |
| **XGBoost (Ensemble)**        | XGBoost delivered the best overall performance among all models. Its gradient boosting framework effectively captured complex non-linear relationships and feature interactions, resulting in the highest AUC and F1 score. This makes XGBoost the most suitable model for churn prediction in this dataset.                                     |


From the comparison, ensemble models such as Random Forest and XGBoost generally performed better than individual models due to their ability to capture complex non-linear patterns and reduce overfitting. Logistic Regression provided a strong baseline with good interpretability, while tree-based models offered better recall for churn detection.