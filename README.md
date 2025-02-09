Justificación taller Git_Hub
1) Análisis de la base de datos.
Según la descripción proporcionada en los anexos de la base de datos, la composición de la base de datos es la siguiente:
*Existen 3 variables categóricas: 1.Season 2.Clima 3.Time_of_day
*Existen 6 variables numéricas: 1.Weekday 2.Temp 3.Atemp 4.Hum 5.Windspeed 6.Cnt


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
