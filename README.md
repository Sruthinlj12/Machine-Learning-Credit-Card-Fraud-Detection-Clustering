💳 Credit Card Fraud Detection & Clustering

This project involves building and evaluating machine learning models to detect fraudulent credit card transactions using classification algorithms, as well as applying clustering techniques to explore patterns in the data.
📂 Dataset

Source: Kaggle - Credit Card Fraud Detection
Description: This dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions.
Attributes: Most features are the result of a PCA transformation due to confidentiality issues. Only Time, Amount, and Class are not PCA-transformed.

🎯 Objective

Detect fraudulent credit card transactions using supervised learning algorithms.
Compare performance metrics such as accuracy, F1-score, precision, and error metrics.
Explore transaction patterns using K-Means and Agglomerative Clustering.

🧠 Algorithms Used
🔍 Supervised Learning

  Logistic Regression
  Decision Tree Classifier
  K-Nearest Neighbors (KNN)
  Each model is evaluated based on:

   Accuracy
   F1 Score
   Precision
   Mean Squared Error (MSE)
   Mean Absolute Error (MAE)
   Confusion Matrix

📊 Unsupervised Learning

   K-Means Clustering
   Agglomerative Clustering
   Visualized using Dendrograms
   Evaluated using Silhouette Score

📈 Data Preprocessing

  Dataset is highly imbalanced. To balance the classes, a sample of 492 legitimate transactions is randomly selected to match the number of fraudulent transactions.
  Feature scaling is done using StandardScaler for clustering.
