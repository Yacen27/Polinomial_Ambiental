
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

# 🔹 Escalamos los datos (importante para Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Entrenamos el modelo Lasso con alpha=0.1
lasso = Lasso(alpha=5)
lasso.fit(X_train_scaled, y_train)

# 🔹 Predicciones
y_pred = lasso.predict(X_test_scaled)

# 🔹 Evaluamos el modelo con MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# 🔹 Mostramos los coeficientes
print("Coeficientes del modelo Lasso:")
print(lasso.coef_)



#################### Con grid search ##############################


from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [1, 2, 3, 4, 5]}
lasso_cv = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Mejor alpha encontrado: {lasso_cv.best_params_['alpha']}")