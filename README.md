Justificación taller Git_Hub
1) Análisis de la base de datos.
Según la descripción proporcionada en los anexos de la base de datos, la composición de la base de datos es la siguiente:
* Existen 3 variables categóricas: 1.Season 2.Clima 3.Time_of_day
* Existen 6 variables numéricas: 1.Weekday 2.Temp 3.Atemp 4.Hum 5.Windspeed 6.Cnt


La descripción de cada variable es la siguiente:
1. Season: Estación del año (Winter, Spring, Summer, Fall)
2. Weekday: Día de la semana (De 0 a 6)
3. Wheatersit: Clima (Clear, Mist, Light Rain, Heavy Rain)
4. Temp: Temperatura
5. Atemp: Sensación de temperatura
6. Hum: Humedad
7. Windspeed: Velocidad del viento
8. Cnt: Cantidad de bicicletas rentadas
9. Time_of_day: Parte del día (Morning, Evening, Night)


* Una vez se conoce la naturaleza de cada variable, es necesario verificar si en el Dataframe existen valores nulos o vacíos, mediante el comando df.isnull().sum() se suman la cantidad de vacíos que se encuentren en el Dataframe, los resultados muestran que no existe ningún vacío en la base de datos. 
* De igual manera, es necesario verificar si existen registros duplicados en el Dataframe, mediante el comando df.duplicated().sum() se suman la cantidad de registros duplicados que se encuentran en el Dataframe, los resultados muestran que existen 42 registros duplicados en la base de datos. Se toma la decisión de eliminar los registros duplicados debido a la optimización y eficiencia computacional, sin mencionar que evita sesgos en los posteriores modelos y reduce el sobreajuste de los mismos.

Una vez realizados estos pasos, se concluye con el paso de exploración y perfilamiento de los datos, junto con la limpieza y preparación de los mismos.


Creación del primer modelo polinomial de orden 1:

Una vez se concluyó la parte de exploración y perfilamiento de datos, se busca terminar de preparar los datos para poder hacer la estimación del modelo de regresión múltiple. Para esto, es necesario dumificar las variables categóricas con el objetivo de que el modelo pueda estimar su relación con la variable independiente de manera correcta. Con el objetivo de corregir el problema de multicolinealidad generado al dumificar estas variables, se opta por eliminar una categoría específica de cada variable, las variables dumificadas y la categoría de referencia se muestran a continuación:

Variables que se van a dumificar:
* Season: Categoría de referencia (Fall)
* Wheatersit: Categoría de referencia (Clear)
* Time_of_day: Categoría de referencia (Evening)

Una vez dumificadas las variables, se procede a estimar el modelo de regresión polinomial de orden 1 por el método de mínimos cuadrados ordinarios o OLS por sus siglas en inglés, los resultados de este modelo muestran un valor de r2 de tan solo 0.43, lo que quiere decir que las variables independientes solamente explican un 43% de la variabilidad de la demanda de bicicletas. Adicionalmente, los p_valores muestran que existen variables que no son relevantes al momento de explicar la demanda de bicicletas. Con p_valores de 0.958 , 0.114 , 0.127 las variables Heavy_Rain , Windspeed y Mist son completamente irrelevantes al momento de explicar la demanda de bicicletas, por esta razón, para el modelo posterior de machine learning se opta por eliminar estas variables.

Variables eliminadas:
* Weather_Heavy_Rain
* Windspeed
* Weather_Mist

Al evaluar las métricas del modelo polinomial de orden 1 con las variables irrelevantes eliminadas. Se obtienen los siguientes resultados:
* R2: 0.42
* RMSE: 139.85
* MAE: 103.7

Como se puede observar, las variables independientes del modelo son capaces de explicar el 42% de la variabilidad de la demanda de bicicletas, con relación a la capacidad predictiva del modelo, al calcular estas dos métricas como % de la media de la variable y (Demanda bicicletas) y obtener porcentajes de 73.65% y 54.62% respectivamente los cuales son superiores al 20%, se establece que el modelo tiene una baja capacidad predictiva y presenta errores altos. Se concluye en base a todas las métricas evaluadas que el modelo en términos generales es muy malo, tanto prediciendo como explicando la variabilidad de la variable dependiente.

* RMSE % Media: 73.65%
* MAE % Media: 54.62%
