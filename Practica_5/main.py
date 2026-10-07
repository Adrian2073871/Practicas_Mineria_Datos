import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os
from Practica_3.Graficas.graficas import guardar_figura
from Utils.lector_dataset import read_dataset
from Utils.CreadorCarpetas import crear_carpeta
from .modelo_regresion import *
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import linregress

CARPETA_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
if __name__ == "__main__":
    df = read_dataset()
    if df is not None:
        df_locales = df[["home_shots_on_target", "home_goals_full_time"]]
        df_visitantes = df[["away_shots_on_target", "away_goals_full_time"]]
        data = []
        data.append(df_locales)
        data.append(df_visitantes)
        equipos = ["locales", "visitantes"]
        variables_explicativas = ["home_shots_on_target", "away_shots_on_target"]
        variables_objetivo = ["home_goals_full_time", "away_goals_full_time"]
        metrica = ["R^2", "MAE", "RMSE"]
        crear_carpeta(CARPETA_IMG)
        for dataset, equipo, variable_explicativa, variables_objetivo in zip(data, equipos, variables_explicativas, variables_objetivo):
            entrenamiento = []
            prueba = []
            X = df[[variable_explicativa]]
            Y = df[variables_objetivo]
            #Crear grafico de dispersion
            fig = plt.figure(figsize=(12, 10))
            sns.regplot(data=dataset, x=variable_explicativa, y = variables_objetivo)
            ruta = guardar_figura(fig, os.path.join(CARPETA_IMG, f"diagrama_dispersion_{equipo}.png"))
            print(f"Grafica guardada: {ruta}")
            modelo = crear_modelo_regresion_lineal()
            #dividir los datos en prueba y entrenamiento
            x_train, x_test, y_train, y_test = dividir_datos(X, Y)
            #Entrenar el modelo
            modelo.fit(x_train, y_train)
            #Predicciones
            y_pred_test = modelo.predict(x_test)
            y_pred_train = modelo.predict(x_train)
            #Calculo de metricas en ambos conjuntos de entrenamiento y prueba
            #Calculo de R_cuadrada 
            entrenamiento.append(r2_score(y_train, y_pred_train))
            prueba.append(r2_score(y_test, y_pred_test))
            #Calculo de MAE
            entrenamiento.append(mean_absolute_error(y_train, y_pred_train))
            prueba.append(mean_absolute_error(y_test, y_pred_test))
            #Calculo de MSE
            entrenamiento.append(root_mean_squared_error(y_train, y_pred_train))
            prueba.append(root_mean_squared_error(y_test, y_pred_test))
            #Diccionario que contiene las metricas
            metricas = dict()
            metricas["Metrica"] = metrica
            metricas["Entrenamiento"] = entrenamiento
            metricas["Prueba"] = prueba
            tabla_metricas = pd.DataFrame(metricas)
            print("\nResultados del modelo de regresion lineal para equipos: ", equipo)
            print(tabla_metricas)
            print("\n\n")
    else:
        print("\nLos datos no se pudieron cargar")







