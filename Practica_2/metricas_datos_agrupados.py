import pandas as pd
from pandas import DataFrame
def crear_metricas_datos_agrupados(columna_a_agrupar:str, categoria:str, df:DataFrame, metricas:list)->DataFrame:
    gdf = df.groupby(columna_a_agrupar)[categoria].agg(metricas)
    return gdf