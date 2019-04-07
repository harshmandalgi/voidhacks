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
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

@app.route('/predict_dt', methods=['POST', 'GET'])
def predict():

	# Importing the dataset
	dataset = pd.read_csv('Clusters.csv')
	dataset.describe()
	X=dataset.iloc[:,1:6].values
	Y=dataset.iloc[:,6:].values


	# Splitting the dataset into the Training set and Test set
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

	# Feature Scaling
	sc_X = StandardScaler()
	sc_Y=StandardScaler()
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.transform(X_test)
	Y_train = sc_Y.fit_transform(Y_train)
	Y_test = sc_Y.transform(Y_test)
	print(Y_test.shape)

	# Fitting Decision Tree Regression to the dataset
	regressor = DecisionTreeRegressor(random_state = 0)
	regressor.fit(X_train, Y_train)

	# Predicting a new result
	Y_pred = regressor.predict([Y_test[0]])

	# Visualising the Decision Tree Regression results (higher resolution)
	X_grid = (np.arange(min(X[0]), max(X[0]), 0.01))
	X_grid = X_grid.reshape((len(X_grid), 1))
	plt.scatter(X, Y, color = 'red')
	plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
	plt.title('(Decision Tree Regression)')
	plt.xlabel('No of Cases')
	plt.ylabel('Probablity of Deaths')
	plt.show()

	return Y_pred

if __name__ == "__main__":
    app.run(debug=True)