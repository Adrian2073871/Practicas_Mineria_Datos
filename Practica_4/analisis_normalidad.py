""""
Este modulo dentro de la practica 4, es utilizado para realizar un analisis de normalidad
con el objetivo de recabar evidencia suficiente para saber si el conjunto de datos
analizado sigue una distribucion normal o no. El analisis de normalidad sera util
para decidir si el analisis se debe realizar mediante un analisis ANOVA o una prueba de Kruskal-Wallis.
"""
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot
import pandas as pd
def generar_modelo(relacion_variables, df_analizar):
    try:
        modelo = smf.ols(relacion_variables, data=df_analizar).fit()
        return modelo
    except Exception:
        print("Ocurrio un error al entrenar el modelo")
        return None

def ejecutar_test_shapiro(residuos):
    if residuos is not None:
        stat, p = stats.shapiro(residuos)
        return stat, p
    return None
def ejecutar_prueba_levene(grupo1, grupo2, grupo3):
    stat, p = stats.levene(grupo1, grupo2, grupo3, center='median', proportiontocut=0.05)
    return stat, p

def ejecutar_prueba_wallis(grupo1, grupo2, grupo3):
    stat, p = stats.kruskal(grupo1, grupo2, grupo3)
    return stat, p

def reporte_metricas(stat, p, nombre_prueba):
    print(nombre_prueba)
    print(f"\nEstadistico W: {stat}")
    print(f"\nP-Valor: {p}")




    