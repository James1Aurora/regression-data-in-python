import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#melihat lima baris pertama dari data
dataset = pd.read_csv('/data_penghasilan.csv')
dataset.head()

#melihat jumlah baris dan kolom pada data
print(dataset.shape)

#membuat training dan test data
from sklearn.model_selection import train_test_split

X = dataset.iloc[:, :-1].values #kolom lama_bekerja
y = dataset.iloc[:, :-1].values #kolom gaji

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#jumlah data X train dan y test
print(len(X_train), len(y_test))

#training model pada dataset
#training model simple linear regression pada training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
