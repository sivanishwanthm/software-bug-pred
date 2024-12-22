#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (4, 3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve, auc, confusion_matrix, f1_score,precision_score,recall_score
from sklearn.cluster import KMeans
import umap
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

from sklearn.neighbors import KNeighborsClassifier
import umap.umap_ as umap
from sklearn.model_selection import train_test_split


# In[2]:


a = pd.read_csv(r"C:\Users\sivan\Downloads\soft_engg_project\Data\eclipse\single-version-ck-oo.csv")
a


# In[3]:


## We have delimited file. splitting the columns based on the demiliter ';' ##
## we use r before reading a csv file to read in as a raw text ignoring the / ##
eclipse_jdt = pd.read_csv(r"C:\Users\sivan\Downloads\soft_engg_project\Data\eclipse\single-version-ck-oo.csv",delimiter=';')
eclipse_pdt = pd.read_csv(r"C:\Users\sivan\Downloads\soft_engg_project\Data\pde\pde\single-version-ck-oo.csv",delimiter=';')
mylyn = pd.read_csv(r"C:\Users\sivan\Downloads\soft_engg_project\Data\mylyn\single-version-ck-oo.csv",delimiter=';')
lucene = pd.read_csv(r"C:\Users\sivan\Downloads\soft_engg_project\Data\lucene\single-version-ck-oo.csv",delimiter=';')
equinox = pd.read_csv(r"C:\Users\sivan\Downloads\soft_engg_project\Data\equinox\single-version-ck-oo.csv",delimiter=';')


# In[4]:


# Here we clean the data by dropping NaNs and Nulls 
eclipse_jdt.dropna(axis = 1, inplace=True)
eclipse_pdt.dropna(axis = 1, inplace=True)
equinox.dropna(axis = 1, inplace=True)
lucene.dropna(axis = 1, inplace=True)
mylyn.dropna(axis = 1, inplace=True)

# All NaNs are dropped
print("NaNs in eclipse_jdt", np.sum(np.sum(eclipse_jdt.isna(), axis=0))) 
print("NaNs in eclipse_pdt", np.sum(np.sum(eclipse_pdt.isna(), axis=0)))
print("NaNs in equinox", np.sum(np.sum(equinox.isna(), axis=0)))
print("NaNs in lucene", np.sum(np.sum(lucene.isna(), axis=0)))
print("NaNs in mylyn", np.sum(np.sum(mylyn.isna(), axis=0)))


# In[5]:


## we dont have  Nulls in the dataframes ##
print("nulls in eclipse_jdt", np.sum(np.sum(eclipse_jdt.isnull(), axis=0))) 
print("nulls in eclipse_pdt", np.sum(np.sum(eclipse_pdt.isnull(), axis=0)))
print("nulls in equinox", np.sum(np.sum(equinox.isnull(), axis=0)))
print("Nulls in lucene", np.sum(np.sum(lucene.isnull(), axis=0)))
print("Nulls in mylyn", np.sum(np.sum(mylyn.isnull(), axis=0)))


# In[6]:


eclipse_jdt.head()


# In[7]:


## We concats all the 5 dfs into a single df ##
df = pd.concat([eclipse_jdt, eclipse_pdt, equinox, lucene, mylyn], ignore_index=True)

## We are removing extra space in the columns # 
df.columns = df.columns.str.replace(' ', '')

print("Full dataframe shape:",df.shape, '\n')
print("Predictors:")

## These are input columns ##
for name in df.columns.values[1:18].tolist():
    print(name, end=',')
    
## Output col ##
print("\n\nPredictable:", df.columns.values[18])
df.head()


# In[8]:


df.columns


# In[9]:


# Shuffle data before removing classname
df = df.drop("classname",axis=1)

# dont need class name;
X = df.iloc[:, 1:-6]
y = df["bugs"]

print("X:", X.shape)
print("y:", y.shape)


# In[10]:


X.head()


# In[11]:


y = y.apply(lambda x: 2 if x > 2 else x)
y.unique()


# In[12]:


print("X:", X.shape)
print("y:", y.shape)


# In[13]:


# Basically checking for unique values in the y #
unique, counts = np.unique(y, return_counts=True)
print("Classes:", unique.tolist())
print("Counts:", counts.tolist())

plt.bar(unique, counts, color=['g', 'blue', 'r'], alpha=0.8)
plt.title("Bugs and Appearance")
plt.xticks(range(len(unique)))
plt.ylabel("Appearance")
plt.xlabel("Bugs");


# In[14]:


#Line Plot
unique, counts = np.unique(y, return_counts=True)

print("Classes:", unique.tolist())
print("Counts:", counts.tolist())

plt.plot(unique, counts, marker='o', linestyle='-', color='b')
plt.title("Bugs and Appearance")
plt.xticks(unique)
plt.ylabel("Appearance")
plt.xlabel("Bugs")
plt.grid(True)
plt.show()


# In[15]:


# Here we divide the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("Train:", X_train.shape, y_train.shape,
      "Test:", X_test.shape, y_test.shape)


# In[16]:


## Scaling the features
X_train_scaled = pd.DataFrame(StandardScaler().fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(StandardScaler().fit_transform(X_test.values), columns=X_test.columns, index=X_test.index)

data_for_viz = X_train_scaled.copy()


# In[17]:


data_for_viz.head()


# In[18]:


# Here we use UMAP for dimensionality reduction.
reducer = umap.UMAP(verbose=False)
embedding = reducer.fit_transform(X_train_scaled)
embedding.shape


# In[19]:


embedding = pd.DataFrame(embedding)
embedding.head()


# In[20]:


# predicting the preds_kmeans_umap
kmeans_umap = KMeans(n_clusters=3)
kmeans_umap.fit(embedding)
preds_kmeans_umap = kmeans_umap.predict(embedding)

fig, ax = plt.subplots(figsize=(8,4))

plt.scatter(embedding[0], embedding[1], c=preds_kmeans_umap, s=20, cmap='viridis')
cbar = plt.colorbar(boundaries=np.arange(4)-0.5)
cbar.set_ticks(np.arange(3))
cbar.set_ticklabels(np.unique(preds_kmeans_umap))

plt.scatter(kmeans_umap.cluster_centers_[:, 0], kmeans_umap.cluster_centers_[:, 1], c='red', s=150, alpha=0.7);

plt.title('Bugs, Software Properties & K-Means Centers', fontsize=14);


# In[21]:


acc = accuracy_score(preds_kmeans_umap, y_train)
print("K-Means Accuracy Score:", acc)


# In[22]:


# predicting y_train and kmeans centers
fig, ax = plt.subplots(figsize=(8,4))

plt.scatter(embedding[0], embedding[1], c=y_train, s=20, cmap='viridis')
cbar = plt.colorbar(boundaries=np.arange(4)-0.5)
cbar.set_ticks(np.arange(3))
cbar.set_ticklabels(np.unique(y_train))

plt.scatter(kmeans_umap.cluster_centers_[:, 0], kmeans_umap.cluster_centers_[:, 1], c='red', s=150, alpha=0.7);

plt.title('Bugs & Software Properties & K-Means Centers | 2D - UMAP', fontsize=14);


# In[23]:


## Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[24]:


# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[25]:


# Get the correlation matrix for the features
correlation_matrix = X.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()


# In[26]:


# KNN Algorithm ##

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a KNN classifier with a specified number of neighbors (e.g., 5)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the KNN classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Calculate the accuracy of the KNN model


accuracy = accuracy_score(y_test, y_pred)
print("KNN Classifier Accuracy:", accuracy)


# In[27]:


# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[28]:


conf_matrix = confusion_matrix(y_test, y_pred)

# Convert the confusion matrix to a DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=sorted(set(y_test)), columns=sorted(set(y_test)))

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, cmap='Blues', fmt='g', linewidths=.5)
plt.title("Confusion Matrix")
plt.show()

