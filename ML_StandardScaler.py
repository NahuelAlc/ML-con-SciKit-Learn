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

from sklearn.preprocessing import StandardScaler
X_new = StandardScaler().fit_transform(X)

#Mostramos el dataset por pantalla
plt.scatter(X_new[:,0], X_new[:,1], c=y);
plt.show()