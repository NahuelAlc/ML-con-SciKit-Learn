import numpy as np
import matplotlib.pylab as plt


#Generamos datos

x = np.random.exponential(10, (1000)) + np.random.normal(0, 1, (1000))

"""
Centramos en 0 los datos y dividimos entre la desviación tipica. Ahora los datos están expresados
en cuantas desviaciones tipicas nos alejamos de la media.
"""

plt.hist((x - np.mean(x))/np.std(x), 30)
plt.show()
