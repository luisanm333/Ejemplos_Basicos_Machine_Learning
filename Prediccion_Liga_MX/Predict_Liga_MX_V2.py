# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:35:33 2023

@author: luis_
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Liga_MX_OK.csv', index_col = 0)

[Fil, Col] = df.shape

#Crea nuevo Data Frame para ordenar los datos
match = pd.DataFrame(columns=['Torneo','Jornada','Fecha','Dia','Hora','Equipo1',
                              'Goles_Eq1','Goles_Eq2','Equipo2','Resultado','Sede'])


Equipos = df['Equipo1'].unique().tolist() # Lista de equipos sin repetir


"""
##################### #####################
####        Limpiando Datos           #####
######################## ##################
"""

# Reacomodando datos en el Data Frame "match"
# para tener los encuentros separados por equipo

cnt = 1
for j in Equipos:
    for i in range(1,Fil+1,1): # i es la fila
        # Si es local
        if df.loc[i,'Equipo1'] == j: # j es el equipo
            match.loc [cnt,:] = df.loc[i,:]
            if (match.loc[cnt,"Goles_Eq1"] > match.loc[cnt,"Goles_Eq2"]):
                match.loc [cnt,'Resultado'] = "G"
            elif (match.loc[cnt,"Goles_Eq1"] < match.loc[cnt,"Goles_Eq2"]):
                match.loc [cnt,'Resultado'] = "P"
            else:
                match.loc [cnt,'Resultado'] = "E"
            match.loc [cnt,'Sede'] = 'Local'
            
            cnt = cnt + 1
        # Si es visitante    
        elif df.loc[i,'Equipo2'] == j: # j es el equipo
            match.loc [cnt,:] = df.loc[i,:]
            match.loc [cnt,'Equipo1'] = df.loc[i,'Equipo2']
            match.loc [cnt,'Goles_Eq1'] = df.loc[i,'Goles_Eq2']
            match.loc [cnt,'Goles_Eq2'] = df.loc[i,'Goles_Eq1']
            match.loc [cnt,'Equipo2'] = df.loc[i,'Equipo1']
            if (match.loc[cnt,"Goles_Eq1"] > match.loc[cnt,"Goles_Eq2"]):
                match.loc [cnt,'Resultado'] = "G"
            elif (match.loc[cnt,"Goles_Eq1"] < match.loc[cnt,"Goles_Eq2"]):
                match.loc [cnt,'Resultado'] = "P"
            else:
                match.loc [cnt,'Resultado'] = "E"
            match.loc [cnt,'Sede'] = 'Visita'
            
            cnt = cnt + 1


#Comporbando la cantidad de partidos que jugó cada equipo
partidos = match["Equipo1"].value_counts()

# Comporbando la cantidad de partidos que se jugaron cada semana
p_semana = match["Jornada"].value_counts()

# Limpiando los datos para Machine Learning
# Machine Learning trabaja con datos numéricos
match_type = match.dtypes

#convirtiendo la columna Fecha a tipo datetime
match["Fecha"] = pd.to_datetime(match["Fecha"],utc=False)
match_type = match.dtypes

match["Jornada"] = pd.to_numeric(match["Jornada"])
match["Goles_Eq1"] = pd.to_numeric(match["Goles_Eq1"])
match["Goles_Eq2"] = pd.to_numeric(match["Goles_Eq2"])
match_type = match.dtypes

# Creando categorías de algunas variables para usarlas como entradas en el modelo

# Creando un código para visita o local
# Si es local, el código es 0, si es visitante, el código es 1
# convierte de strings a categorías y luego a entero
match["Sede_codigo"] = match["Sede"].astype("category").cat.codes

#Creando código único para cada oponente
match["Equipo2_codigo"] = match["Equipo2"].astype("category").cat.codes

# Convirtiendo la hora a entero (sólo nos quedamos con la hora sin los minutos)
match["Hora"] = match["Hora"].str.replace(":.+","",regex=True).astype("int")

# Creando un código para el día de la semana
# Lunes = 0, Martes = 1, Miercoles =2, ..., Domingo = 6
match["Dia_codigo"] = match["Fecha"].dt.dayofweek

# Creando el Target
# El target será 1 si el equipo gana, 0 si pierde o empata
match["target"] = (match["Resultado"] == "G").astype("int") #match["Resultado"].astype("category").cat.codes

"""
#####################      #####################
####  Modelo inicial de Machine Learning  #####
#####################    #######################
"""

# Utilizamos el modelo Random Forest 
# este modelo puede detectar no linealidades en los datos

# Ex. El código del quipo 11, no implica que ese equipo tenga mayor peso 
# (dificultad) que el equipo con el código 3

# El modelo Random Forest es una serie de "decision trees" (arboles de decision),
# donde cada decision tree tiene parámetros ligeramente diferentes.
# El modelo Random Forest tiene muchos parámeros aleatorios

from sklearn.ensemble import RandomForestClassifier

# n_estimators:       Es el número de decision trees que queremos entrena.
#                    Si n_estimator es mayor, el modelo será más preciso, 
#                    pero tardará más en entrenar
# min_samples_split: Es el número de muestras que queremos tener en una hoja
#                    del decision tree antes de que dividamos el nodo.
#                    Si min_samples_split es mayor, es menos probable que 
#                    sobreajustemos (overfit), pero la precision del modelo 
#                    disminuye
# random_state:      Asignamos valor 1 para tener repetitibilidad, siempre que 
#                    los datos sean los mismos
#Los resultados están en la matriz marged

model_rf = RandomForestClassifier(n_estimators = 60, min_samples_split = 10, random_state = 1)

# Separando datos de entrenamiento y datos de prueba
# Como tenemos datos en series de tiempo, debemos tener cuidado al separalos
# ya que no queremos entrenar con datos futuros para hacer predicciones del pasado
# es decir, todos los datos de prueba debes ser posteriores a los de entrenamiento

fecha = '2023-11-9'
train = match[match["Fecha"] < fecha]
test = match[match["Fecha"] > fecha]

# Entradas del modelo
predictors = ["Sede_codigo", "Equipo2_codigo", "Hora", "Dia_codigo"]

# Entrenando el modelo Random Forest
model_rf.fit(train[predictors], train["target"])


"""
Predicción de datos con el modelo ya entrenado
"""
predictions = model_rf.predict(test[predictors]) #haciendo la predicción de todos los datos

"""
Evaluando la precision del modelo
"""

# accuracy_score indica qué porcentaje de las predicciones fueron acertadas
from sklearn.metrics import accuracy_score

acc = accuracy_score(test["target"], predictions)

# Matriz de confusión
#Crea nuevo Data Frame para ordenar los datos
comb = pd.DataFrame(dict(actual = test["target"], prediction = predictions))
matrix = pd.crosstab(index = comb["actual"], columns = comb["prediction"])

# Este modelo no hace buenas predicciones si gana, sólo acierta cuando empata o pierde

# Usando otra métrica para evaluar el modelo

# precision_score: cuando predecimos que Gana, precision_score dice qué 
#                  porcentaje de las veces el equipo ganó
from sklearn.metrics import precision_score

prec = precision_score(test["target"], predictions)

test.to_csv('test_name.csv')

"""
    Mejorando con Rolling Averages
"""

# Agregando más predictores
# Considerando el resultado de los últimos 4 partidos
# Considerando goles a favor y goles en contra

# Creando grupos de partidos por equipo
grouped_matches = match.groupby("Equipo1") 

# Ordena por fecha
group = grouped_matches.get_group("America").sort_values("Fecha")


# Función para ordenar los equiupos por fecha en que se jugó el partido
# se agregan dos nuevas columnas con el promedio de los últimos 4 juegos
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Fecha")
    rolling_stats = group[cols].rolling(6, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)    
    return group

cols = ["Goles_Eq1", "Goles_Eq2"]

new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

matches_rolling = match.groupby("Equipo1").apply(lambda x: rolling_averages(x, cols, new_cols))

matches_rolling = matches_rolling.droplevel('Equipo1')

matches_rolling.index = range(matches_rolling.shape[0])


# Calculando los días que han pasado desde su ultimo partido
# La resta de dos fechas entrega un tipo de dato timedelta, 
# al cual se le puede extraer los días, min, seg, etc
[Fil_mr, Col_mr] = matches_rolling.shape

for i in range(0,Fil_mr,1):
    if   matches_rolling.loc [i,"Fecha"] <=  matches_rolling.loc [0,"Fecha"]:
        matches_rolling.loc [i,"Dias_ultimo_Partido"] = (matches_rolling.loc[i,"Fecha"]
                                                         - matches_rolling.loc[i,"Fecha"]).days # Obteniendo únicamente el número de días
    else:
        matches_rolling.loc [i,"Dias_ultimo_Partido"] = (matches_rolling.loc[i,"Fecha"]
                                                         - matches_rolling.loc[i-1,"Fecha"]).days

matches_rolling_type = matches_rolling.dtypes

matches_rolling["Dias_ultimo_Partido"] = matches_rolling["Dias_ultimo_Partido"]
#match["Hora"] = match["Hora"].str.replace(":.+","",regex=True).astype("int")

new_cols_2 = ["Dias_ultimo_Partido"]

def make_predictions(data, predictors):
    train = data[data["Fecha"] < fecha]
    test = data[data["Fecha"] > fecha]
    model_rf.fit(train[predictors], train["target"])
    preds = model_rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    prec_model2 = precision_score(test["target"], preds)
    acc2 = accuracy_score(test["target"], preds)
    return combined, prec_model2, acc2

combined, prec_model2,acc2 = make_predictions(matches_rolling, predictors + new_cols + new_cols_2)

combined = combined.merge(matches_rolling[["Fecha", "Equipo1", "Equipo2", "Resultado"]], left_index=True, right_index=True)

"""
    Combinando resultados de Local y Visitante
"""

merged = combined.merge(combined, left_on=["Fecha", "Equipo1"], right_on=["Fecha", "Equipo2"])
# merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()
