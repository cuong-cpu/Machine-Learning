import numpy as np 
import pandas as pd 

df = pd.read_csv('iris.csv',header= None)
df.head()
print(df)

print(df)

df[2].to_csv('new_data.csv')