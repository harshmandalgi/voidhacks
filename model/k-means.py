from flask import Flask, render_template, url_for, request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

app = Flask(__name__)
#api = Api(app)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
	# Importing the dataset
	dataset = pd.read_csv('Clusters.csv')
	dataset.describe()
	X=dataset.iloc[:,1:6].values
	Y=dataset.iloc[:,6:].values


	# Splitting the dataset into the Training set and Test set
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

	# Feature Scaling
	sc_X = StandardScaler()
	sc_Y = StandardScaler()
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.transform(X_test)
	Y_train = sc_Y.fit_transform(X_train)
	Y_test = sc_Y.transform(X_test)

	# Using the elbow method to find the optimal number of clusters
	wcss = []
	for i in range(1, 11):
	    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 50)
	    kmeans.fit(X)
	    wcss.append(kmeans.inertia_)
	plt.plot(range(1, 11), wcss)
	plt.title('The Elbow Method')
	plt.xlabel('Number of clusters')
	plt.ylabel('WCSS(Within Clusters Sum of Squares)')
	plt.show()

	# Fitting K-Means to the dataset
	kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
	y_kmeans = kmeans.fit_predict(X)

	# Visualising the clusters
	plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Acute Diarrhoeal Diseases')
	plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Malaria')
	plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Acute Respiaratory Infection ')
	plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Japanese Encephalitis')
	plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Viral Hepatitis')
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 80, c = 'yellow', label = 'Centroids')
	plt.title('Clusters of Percentage of Deaths')
	plt.ylabel('Probablity of Deaths')
	plt.xlabel('No of Cases')
	plt.legend()
	plt.show()

	return y_kmeans

if __name__ == "__main__":
    app.run(debug=True)