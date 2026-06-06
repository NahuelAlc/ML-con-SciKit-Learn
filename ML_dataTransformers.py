#imports
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

#Vamos a predecir z en función de x e y.

#Cargamos el dataset en una variable
df = pd.read_csv("drawndata1.csv")
print(df.head(3))

#Cargamos columnas x e y en matriz 2D
X= df[['x', 'y']].values

#calculamos true o false en el vector target
y = df['z'] == "a"

#Escalamos los datos para que no tenga más influencia X que y
#La forma estandar de escalarlos es con StandardScaler()

from sklearn.preprocessing import StandardScaler, QuantileTransformer

#pipeline
from sklearn.pipeline import Pipeline

#usando StandardScaler() 
#X_new = StandardScaler().fit_transform(X)

#mapeo de los datos entre 0 y 1
#X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)

#Mostramos el dataset por pantalla
#plt.scatter(X_new[:,0], X_new[:,1], c=y);
#plt.show()

from sklearn.neighbors import KNeighborsClassifier

#Pintado de puntos antes y después de transformar + predicción.
def plot_output(scaler):
    pipe = Pipeline([
        ("scale", scaler),
        ("model", KNeighborsClassifier(n_neighbors=20, weights='distance'))
    ])

    pred = pipe.fit(X, y).predict(X)

    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Original Data")
    plt.subplot(132)
    X_tfm = scaler.transform(X)
    plt.scatter(X_tfm[:, 0], X_tfm[:, 1], c=y)
    plt.title("Transformed Data")
    plt.subplot(133)
    X_new = np.concatenate([
        np.random.uniform(0, X[:, 0].max(), (5000, 1)), 
        np.random.uniform(0, X[:, 1].max(), (5000, 1))
    ], axis=1)
    y_proba = pipe.predict_proba(X_new)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=y_proba[:, 1], alpha=0.7)
    plt.title("Predicted Data")

#Los 2 modelos. Uno con StandardScaler y otro con QuantileTransformer

plot_output(scaler=StandardScaler())
#plot_output(scaler=QuantileTransformer(n_quantiles=100))

plt.show()