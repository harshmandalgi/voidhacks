from flask import Flask, render_template, url_for, request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/predict_lr', methods=['POST', 'GET'])
def predict():
	# Importing the dataset
	dataset = pd.read_csv('Clusters.csv')
	dataset.describe()
	X = dataset.iloc[:,1:6].values
	Y = dataset.iloc[:,6:].values

	#Taking care of missing data
	#Imputer is used to fill missing data, represented by 'Nan'
	imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)  #axis=0 ~ Col and axis=1 ~ Row
	imputer = imputer.fit(X[:, 1:])
	imputer = imputer.fit(Y[:, 0:])
	X[:, 0:] = imputer.transform(X[:, 0:])
	Y[:, 0:] = imputer.transform(Y[:, 0:])



	# Splitting the dataset into the Training set and Test set
	X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=0)

	# Feature Scaling
	sc_X = StandardScaler()
	sc_Y=StandardScaler()
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.transform(X_test)
	Y_train = sc_Y.fit_transform(Y_train)
	Y_test = sc_Y.transform(Y_test)


	# Fitting Simple Linear Regression to the Training set
	regressor = LinearRegression()
	regressor.fit(X_train, Y_train)

	# Predicting the Test set results
	y_pred = regressor.predict(Y_test)

	# Visualising the Training set results
	plt.scatter(X_train, Y_train, color = 'red')
	plt.plot(X_train, regressor.predict(X_train), color = 'blue')
	plt.title('Cases vs Diseases (Training set)')
	plt.xlabel('Cases')
	plt.ylabel('Diseases')
	plt.show()

	# Visualising the Test set results
	plt.scatter(X_test, Y_test, color = 'red')
	plt.plot(X_train, regressor.predict(X_train), color = 'blue')
	plt.title('Cases vs Diseases (Test set)')
	plt.xlabel('Cases')
	plt.ylabel('Diseases')
	plt.show()

	return y_pred

if __name__ == "__main__":
    app.run(debug=True)