#imports
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing


#importar datos
X, y = fetch_california_housing(return_X_y=True) 


pipe = Pipeline([
    ("Scale", StandardScaler()),
    ("Model", KNeighborsRegressor())
    ])

pipe.fit(X, y)
pred = pipe.predict(X)

plt.scatter(y, pred)
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.show()