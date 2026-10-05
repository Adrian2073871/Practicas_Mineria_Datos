from Utils.lector_dataset import read_dataset
from Utils.CreadorCarpetas import crear_carpeta
from Practica_3.Graficas.graficas import generar_boxplot
from Practica_3.Graficas.graficas import guardar_figura
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import seaborn as sn
from .analisis_normalidad import *
import os

CARPETA_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
if __name__ == "__main__":
    df = read_dataset()
    if df is not None:
        alpha = 0.05    #Nivel de significancia utilizado para este analisis
        #Separacion de la variable dependiente e independiente en grupos
        df_analizar = df[["home_shots_on_target", "full_time_result"]]
        print(df.groupby("full_time_result")["home_shots_on_target"].describe())
        print("\n\n")
        grupo_H = df_analizar[df_analizar["full_time_result"] == "H"]
        grupo_A = df_analizar[df_analizar["full_time_result"] == "A"]
        grupo_D = df_analizar[df_analizar["full_time_result"] == "D"]
        crear_carpeta(CARPETA_IMG)
        fig = generar_boxplot(df_analizar, "home_shots_on_target", "Diagrama de Bigote", "full_time_result")
        ruta = guardar_figura(fig, os.path.join(CARPETA_IMG, "boxplot.png"))
        print(f"Grafica guardada: {ruta}")
        #En el modelo, se define full time result como variable independiente y categorica
        modelo = generar_modelo("home_shots_on_target ~ C(full_time_result)", df_analizar)
        if modelo is not None:
            #Generar grafico QQ para observar si los residuos siguen una distr. normal
            print("\nResiduos de los datos.")
            print("\n\n")
            print(modelo.resid)
            qqplot(modelo.resid, line="45")
            pylab.savefig(os.path.join(CARPETA_IMG, "qqplot.png"))
            #Generar y guardar un grafico histograma para observar si siguen una distr. normal
            fig = plt.figure(figsize=(12, 10))
            sn.histplot(modelo.resid, kde=True)
            ruta = guardar_figura(fig, os.path.join(CARPETA_IMG, "histograma.png"))
            print(f"Grafica guardada: {ruta}")

            #De los resultados del modelo entrenado, se obtienen los residuos de la diferencia entre el promedio de cada grupo y cada valor
            stat, p = ejecutar_test_shapiro(modelo.resid)
            reporte_metricas(stat, p, "Prueba de Shapiro-Wills")
            print("\n")
            if p<=alpha:
                print("\nSe rechaza H0: Existe ev. estadistica para decir que los datos NO provienen de una distr. normal.")
            else:
                print("\nNo se rechaza H0: Existe ev. estadistica para decir que los datos provienen de una distr. normal")
            print("\n\n")
            stat, p = ejecutar_prueba_levene(grupo_H["home_shots_on_target"], grupo_A["home_shots_on_target"], grupo_D["home_shots_on_target"])
            reporte_metricas(stat, p, "Prueba de Levene")
            print("\n")
            if p > alpha:
                print("\nNo se rechaza H0: Las varianzas son iguales.")
            else:
                print("\nSe rechaza H0: Las varianzas NO son iguales.")
            print("\n")
            stat, p = ejecutar_prueba_wallis(grupo_H["home_shots_on_target"], grupo_A["home_shots_on_target"], grupo_D["home_shots_on_target"])
            reporte_metricas(stat, p, "Prueba de Kruskal-Wallis")
            if p<alpha:
                print("\nSe rechaza H0: Existe ev. estadistica para decir que al menos un grupo es diferente de los demas.")
            else:
                print("\nNo se rechaza H0: Existe ev. estadistica para decir que NO existe diferencias entre los grupos.")

            

        


