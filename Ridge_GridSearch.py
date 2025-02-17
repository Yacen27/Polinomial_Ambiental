import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
df = pd.read_csv("C:/Users/boyac/OneDrive/Desktop/Python/Clases_Personalizadas/Taller_GitHub/Bdatos.csv")
Des = pd.read_excel("C:/Users/boyac/OneDrive/Desktop/Python/Clases_Personalizadas/Taller_GitHub/Descripcion.xlsx")

df_SD = df.drop_duplicates(keep='first')

##### Una vez realizados estos pasos, se concluye con el paso de exploración y perfilamiento de los datos, junto con la limpieza y preparación de los mismos.



############ CREACIÓN DEL MODELO POLINOMIAL #####################
########## Dumificación de variables ############################
df_SD_D= pd.get_dummies(df_SD, columns =['season',"weathersit","time_of_day"], drop_first=True)

##### En este caso la transformación a dummies toma como variable de referencia y por lo tanto elimina del DF las siguientes categorías:
    # Season: Fall
    # Weathersit: Clear
    # Time_of_day: Evening

############ CREACIÓN VARIABLES x y #############################

x = df_SD_D.drop(columns= "cnt")
y = df_SD_D["cnt"]


# 🔹 Dividimos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

# 🔹 Definimos un pipeline para la regresión polinómica con Ridge
pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),  # Transformación polinómica
    ("scaler", StandardScaler()),  # Normalización de los datos
    ("ridge", Ridge())  # Modelo Ridge
])

# 🔹 Definimos los valores de alpha a evaluar en GridSearch
param_grid = {'ridge__alpha': [1, 2, 3, 4, 5]}

# 🔹 Aplicamos GridSearchCV para encontrar el mejor alpha
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 🔹 Mejor alpha encontrado
best_alpha = grid_search.best_params_['ridge__alpha']
print(f"Mejor valor de alpha: {best_alpha}")

# 🔹 Predicciones en el conjunto de prueba con el mejor modelo
y_pred = grid_search.best_estimator_.predict(X_test)

print("RMSE: %.2f" % mean_squared_error(y_test, y_pred, squared=False))
print("MAE: %.2f" % mean_absolute_error(y_test, y_pred))
print('R²: %.2f' % r2_score(y_test, y_pred))

