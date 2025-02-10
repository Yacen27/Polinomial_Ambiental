############## TALLER PROBLEMÁTICA AMBIENTAL ##################
import pandas as pd
df = pd.read_csv("C:/Users/boyac/OneDrive/Desktop/Python/Clases_Personalizadas/Taller_GitHub/Bdatos.csv")
Des = pd.read_excel("C:/Users/boyac/OneDrive/Desktop/Python/Clases_Personalizadas/Taller_GitHub/Descripcion.xlsx")



# Según la descripción de las variables, existen 3 variables categóricas en la base de datos, estas son:
# 1.season 2. weathersit 3.time_of_day
# Por esta razón es necesario tener en cuenta que al hacer los modelos hay que aplicar un enfoque de dummies en estas variables.

############ Establecer si hay datos nulos en el Dataframe ##################
print(df.isnull().sum())
# Se concluye que para ninguna variable existen datos nulos

############ Establecer si hay registros duplicados en el Dataframe #########
Duplicados = df[df.duplicated(keep=False)]
print(df.duplicated().sum())
#Se concluyen que sí existen varios registros duplicados (42 exactamente), por lo que es necesario curarlos y eliminarlos.

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

###### EN PRIMERA INSTANCIA SE CORRE EL MODELO DE ORDEN 1 ##############

###### ESTIMACION DEL MODELO POLINOMIAL DE ORDEN 1 ####################
import statsmodels.api as sm

# Agregar intercepto para el modelo de statsmodels
x = sm.add_constant(x)

# Convertir columnas booleanas a tipo int
x = x.astype({col: int for col in x.select_dtypes(include=["bool"]).columns})

# Ajustar el modelo
model = sm.OLS(y, x).fit()

# Mostrar resultados
print(model.summary())


############### Corrección variables x y ###########################
columnas_eliminadas = ["cnt","weathersit_Heavy Rain","windspeed","weathersit_Mist"]
x = df_SD_D.drop(columns= columnas_eliminadas).values
y = df_SD_D["cnt"].values



##### Dividir el conjunto de datos en entrenamiento y testing ##########
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("RMSE: %.2f" % mean_squared_error(y_test, y_pred, squared=False))
print("MAE: %.2f" % mean_absolute_error(y_test, y_pred))
print('R²: %.2f' % r2_score(y_test, y_pred))

######### ¿Como se si el RMSE es bajo o no? #############

media_y = y.mean()
rmse_porcentaje = ( mean_squared_error(y_test, y_pred, squared=False)/ media_y) * 100
mae_porcentaje = ( mean_absolute_error(y_test, y_pred)/ media_y) * 100

print(f"🔹 RMSE como % de la media de y: {rmse_porcentaje:.2f}%")
print(f"🔹 MAE como % de la media de y: {mae_porcentaje:.2f}%")

####### Al tener tanto un % MAE como un %RMSE superiores al 20%, se concluye que el modelo tiene una mala capcidad productiva (Modelo con errores altos)
# Ahora veremos los coeficientes del modelo
print(list(zip(df_SD_D.columns, regressor.coef_)))



############ ¿Cual grado es el mejor modelo? CREACION DE LA GRILLA ###############

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

polynomial_regression = make_pipeline(
    PolynomialFeatures(),
    LinearRegression()
    )
############### CREACIÓN GRILLA ##################

param_grid = {"polynomialfeatures__degree" : [1,2,3]} #### Grados a los que va a elevar el modelo

from sklearn.model_selection import KFold, GridSearchCV

kfold = KFold(n_splits = 10, shuffle=True, random_state= 77)

modelos_grid = GridSearchCV(polynomial_regression, param_grid, cv=kfold, n_jobs=1, scoring = "neg_root_mean_squared_error") 

modelos_grid.fit(x_train, y_train)

mejor_grado = modelos_grid.best_params_
print(mejor_grado)

############ El mejor modelo es el de grado 3 ################

############ metricas de evaluacion mejor modelo ###############

mejor_modelo = modelos_grid.best_estimator_

############ predicciones mejor modelo #########################

y_pred = mejor_modelo.predict(x_test)
y_pred_train = mejor_modelo.predict(x_train)

########### metricas mejor modelo #################################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# RMSE (Error cuadrático medio raíz)
rmse_test = mean_squared_error(y_test, y_pred, squared=False)


# MAE (Error absoluto medio)
mae_test = mean_absolute_error(y_test, y_pred)


# R² (Coeficiente de determinación)
r2_test = r2_score(y_test, y_pred)


# Mostrar resultados
print(f"Mejor grado del polinomio: {modelos_grid.best_params_['polynomialfeatures__degree']}\n")
print(f"🔹 RMSE en test: {rmse_test:.4f}")
print(f"🔹 MAE en test: {mae_test:.4f}")
print(f"🔹 R² en test: {r2_test:.4f}")
