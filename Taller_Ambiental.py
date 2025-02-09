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
