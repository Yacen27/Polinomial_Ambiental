import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV


df = pd.read_csv(
    "C:/Users/boyac/OneDrive/Desktop/Python/Clases_Personalizadas/Taller_GitHub/Bdatos.csv")
Des = pd.read_excel(
    "C:/Users/boyac/OneDrive/Desktop/Python/Clases_Personalizadas/Taller_GitHub/Descripcion.xlsx")


df_SD = df.drop_duplicates(keep='first')

########## Dumificación de variables ############################
df_SD_D = pd.get_dummies(
    df_SD, columns=['season', "weathersit", "time_of_day"], drop_first=True)
############ CREACIÓN VARIABLES x y #############################

columnas_eliminadas = ["cnt", "weathersit_Heavy Rain",
                       "windspeed", "weathersit_Mist"]
x = df_SD_D.drop(columns=columnas_eliminadas).values
y = df_SD_D["cnt"].values
# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

# Escalar las variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Crear pipeline con regresión polinómica
degree = 3  # Grado del polinomio
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)


# Entrenar el modelo
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predicciones
y_pred = model.predict(X_test_poly)

print("RMSE: %.2f" % mean_squared_error(y_test, y_pred, squared=False))
print("MAE: %.2f" % mean_absolute_error(y_test, y_pred))
print('R²: %.2f' % r2_score(y_test, y_pred))

############# P_VALORES (Cuando el grado sea =1) ########################
import statsmodels.api as sm

# Agregar constante para la intersección (bias) del modelo
X_train_poly_const = sm.add_constant(X_train_poly)

# Agregar constante para la intersección (bias) del modelo
X_train_poly_const = sm.add_constant(X_train_poly)

# Ajustar modelo con statsmodels
model_stats = sm.OLS(y_train, X_train_poly_const).fit()

# Mostrar resumen con p-valores
print(model_stats.summary())

###### Las variables irrelevantes son:
# X5: Windspeed
# X9: Weathersit_Heavy Rain
# X11: Weathersit_Mist

############### CREACIÓN GRILLA ##################

# Grados a los que va a elevar el modelo

polynomial_regression = make_pipeline(
    PolynomialFeatures(),
    LinearRegression()
)
param_grid = {"polynomialfeatures__degree": [1, 2, 3]}


kfold = KFold(n_splits=10, shuffle=True, random_state=77)

modelos_grid = GridSearchCV(polynomial_regression, param_grid,
                            cv=kfold, n_jobs=1, scoring="neg_root_mean_squared_error")

modelos_grid.fit(X_train, y_train)

mejor_grado = modelos_grid.best_params_
print(mejor_grado)
