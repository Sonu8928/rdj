import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
dataset = pd.read_csv("Dataset.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Print initial values
print(y)
print(x)
print(dataset)

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

# Encode categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

# Encode the dependent variable
le = LabelEncoder()
y = np.array(le.fit_transform(y))
print(y)

# Display statistics of 'Age'
print(dataset['Age'].describe())

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print(x_test)

# Train a Decision Tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# Make predictions and evaluate the model
x_predict = np.array(x_test[:, :])
print(classifier.predict(x_predict))
accuracy = classifier.score(x_test, y_test)
print(accuracy)
print(y_test)
