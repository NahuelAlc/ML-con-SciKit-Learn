#abstract
"""
Este proyecto aplica machine learning supervisado (regresión)
sobre el dataset de viviendas de California.

Se utiliza un pipeline con escalado de variables y un modelo KNN,
optimizando hiperparámetros mediante validación cruzada (GridSearchCV),
para predecir el valor medio de viviendas en función de variables socioeconómicas y geográficas.
"""

#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


#importar datos
X, y = fetch_california_housing(return_X_y=True) 
#x = datos
#y = target

#pipeline
pipe = Pipeline([
    ("Scale", StandardScaler()),
    ("Model", KNeighborsRegressor())
    ])

#modelo
mod = GridSearchCV(estimator=pipe,
             param_grid={'Model__n_neighbors': [1,2,3,4,5,6,7,8,9,10]},
             cv=3)

#Guardamos predicciones
mod.fit(X, y)

print(pd.DataFrame(mod.cv_results_)) 

best_model = mod.best_estimator_
pred = best_model.predict(X)

#Mostramos gráfica por pantalla
plt.scatter(y, pred)
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.show()
