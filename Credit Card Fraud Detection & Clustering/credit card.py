import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, mean_squared_error, mean_absolute_error, silhouette_score


credit_card_data = pd.read_csv('C:/Users/HP/Desktop/conflict/creditcard.csv')

print(credit_card_data['Class'].value_counts())
print()

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.Amount.describe())
print()
print(fraud.Amount.describe())

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Logistic Regression
model = LogisticRegression(max_iter=5000000)
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data (Logistic Regression): ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data (Logistic Regression): ', test_data_accuracy)

mse_logistic = mean_squared_error(Y_test, X_test_prediction)
f1_score_logistic = f1_score(Y_test, X_test_prediction)
precision_logistic = precision_score(Y_test, X_test_prediction)
mae_logistic = mean_absolute_error(Y_test, X_test_prediction)

print('Mean Squared Error (Logistic Regression):', mse_logistic)
print('F1 Score (Logistic Regression): ', f1_score_logistic)
print('Precision (Logistic Regression): ', precision_logistic)
print('Mean Absolute Error (Logistic Regression):', mae_logistic)

cm = confusion_matrix(Y_test, X_test_prediction)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()

# Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=2)
decision_tree_model.fit(X_train, Y_train)

print()
X_train_prediction_tree = decision_tree_model.predict(X_train)
training_data_accuracy_tree = accuracy_score(Y_train, X_train_prediction_tree)
print('Accuracy on Training data (Decision Tree): ', training_data_accuracy_tree)

X_test_prediction_tree = decision_tree_model.predict(X_test)
test_data_accuracy_tree = accuracy_score(Y_test, X_test_prediction_tree)
print('Accuracy score on Test Data (Decision Tree): ', test_data_accuracy_tree)

f1_score_tree = f1_score(Y_test, X_test_prediction_tree)
precision_tree = precision_score(Y_test, X_test_prediction_tree)
mse_tree = mean_squared_error(Y_test, X_test_prediction_tree)
mae_tree = mean_absolute_error(Y_test, X_test_prediction_tree)

print('Mean Squared Error (Decision Tree):', mse_tree)
print('F1 Score (Decision Tree): ', f1_score_tree)
print('Precision (Decision Tree): ', precision_tree)
print('Mean Absolute Error (Decision Tree):', mae_tree)

cm_tree = confusion_matrix(Y_test, X_test_prediction_tree)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Decision Tree)')
plt.show()

print()
print()

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

X_train_prediction_knn = knn_model.predict(X_train)
training_data_accuracy_knn = accuracy_score(Y_train, X_train_prediction_knn)
print('Accuracy on Training data (KNN): ', training_data_accuracy_knn)

X_test_prediction_knn = knn_model.predict(X_test)
test_data_accuracy_knn = accuracy_score(Y_test, X_test_prediction_knn)
print('Accuracy score on Test Data (KNN): ', test_data_accuracy_knn)

f1_score_knn = f1_score(Y_test, X_test_prediction_knn)
precision_knn = precision_score(Y_test, X_test_prediction_knn)
mse_knn = mean_squared_error(Y_test, X_test_prediction_knn)
mae_knn = mean_absolute_error(Y_test, X_test_prediction_knn)

print('Mean Squared Error (KNN):', mse_knn)
print('F1 Score (KNN): ', f1_score_knn)
print('Precision (KNN): ', precision_knn)
print('Mean Absolute Error (KNN):', mae_knn)

print()
print()

cm_knn = confusion_matrix(Y_test, X_test_prediction_knn)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (KNN)')
plt.show()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

num_clusters = 3  

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)
kmeans_centroids = kmeans.cluster_centers_  
kmeans_silhouette = silhouette_score(scaled_features, kmeans_labels)
print(f"Silhouette Score for K-Means: {kmeans_silhouette}")

agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)
agg_labels = agg_clustering.fit_predict(scaled_features)
agg_silhouette = silhouette_score(scaled_features, agg_labels)
print(f"Silhouette Score for Agglomerative Clustering: {agg_silhouette}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=kmeans_labels, palette='viridis', s=50)
plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], s=200, color='red', marker='X', label='Centroids')  
plt.title("K-Means Clustering with Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=agg_labels, palette='viridis', s=50)
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

linkage_matrix = linkage(scaled_features, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, orientation='top')
plt.title("Dendrogram for Agglomerative Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()