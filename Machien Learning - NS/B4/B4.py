import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = load_iris()
print(df)

#%%

X_train, X_test, y_train, y_test = train_test_split(df.data, df.target, random_state=0)

model = DecisionTreeClassifier()
new_model = model.fit(X_train,y_train)

#%%
new_model.predict(X_test)
#%%
