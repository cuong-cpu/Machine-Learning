from sklearn.datasets import load_iris
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier

dataset = load_iris() 

X_train, X_test, y_train, y_test = train_test_split(dataset.data,dataset.target,random_state=0)

model = DecisionTreeClassifier()
new_model = model.fit(X_train, y_train)

print(new_model.predict(X_test))
