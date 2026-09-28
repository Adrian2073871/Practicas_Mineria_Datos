import os
import sys

# Permite ejecutar tanto con `python -m Practica_3.main` como con `python Practica_3/main.py`
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import pandas as pd
from pandas import DataFrame
from Utils.lector_dataset import read_dataset
from Utils.CreadorCarpetas import crear_carpeta
from Practica_3.Graficas.graficas import *

CARPETA_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def preparar_datos(df: DataFrame) -> DataFrame:
    """Agrega columnas derivadas (no modifica el dataset original)."""
    fechas = pd.to_datetime(df["date"], format="%d/%m/%Y")
    # La temporada arranca en agosto: 27/08/2011 y 15/05/2012 pertenecen a la 2011-12
    anio_inicio = fechas.dt.year.where(fechas.dt.month >= 8, fechas.dt.year - 1)
    return df.assign(
        temporada=anio_inicio.astype(str) + "-" + (anio_inicio + 1).astype(str).str[-2:],
        total_goles=df["home_goals_full_time"] + df["away_goals_full_time"],
        total_faltas=df["home_fouls"] + df["away_fouls"],
        total_amarillas=df["home_yellow_cards"] + df["away_yellow_cards"],
    )


def construir_configuracion(df: DataFrame) -> dict:
    """
    Diccionario con los parametros de cada grafica.
        clave  -> tipo de grafica
        valor  -> lista de tuplas; cada tupla son los argumentos (en orden) de la funcion del grafico.
    """
    # ---- Datos agrupados / auxiliares ----
    # Promedio de tiros y goles por equipo cuando juega de local
    por_equipo = df.groupby("home_team").agg(
        tiros_prom=("home_shots", "mean"),
        goles_prom=("home_goals_full_time", "mean"),
    ).reset_index()

    # Reparto de resultados finales
    resultados = df["full_time_result"].map({"H": "Local", "D": "Empate", "A": "Visitante"}).value_counts()

    # Victorias como local: 6 equipos con mas victorias y el resto agrupado
    victorias_local = df.loc[df["full_time_result"] == "H", "home_team"].value_counts()
    top_victorias = victorias_local.head(6)
    top_victorias["Otros"] = victorias_local.iloc[6:].sum()

    # Los mapas de calor usan solo las variables int64 originales (sin las columnas derivadas)
    df_original = df.drop(columns=["total_goles", "total_faltas", "total_amarillas"])

    # Bloques del heatmap cruzado: variables del local (filas) y variables del visitante (columnas)
    enteros = df_original.select_dtypes(include="int64").columns
    cols_local = [c for c in enteros if c.startswith("home_")]
    cols_visita = [c for c in enteros if c.startswith("away_")]

    return {
        "heatmap": [
            # (df, titulo, filas, columnas)
            (df_original, "Correlacion entre todas las variables numericas (int64)", None, None),
            (df_original, "Correlacion cruzada: estadisticas del local vs. visitante", cols_local, cols_visita),
        ],
        "scatter": [
            # (X, Y, df, titulo, hue, etiqueta, linea)
            ("home_shots", "home_shots_on_target", df,
             "Tiros vs. tiros a puerta del local segun el resultado", "full_time_result", None, False),
            ("tiros_prom", "goles_prom", por_equipo,
             "Promedio de tiros vs. goles por equipo (como local)", None, "home_team", True),
        ],
        "histograma": [
            # (columna, df, titulo, bins, discrete)
            ("home_corners", df, "Distribucion de tiros de esquina del equipo local", 20, False),
            ("total_faltas", df, "Distribucion de faltas totales por partido", 20, False),
            ("total_goles", df, "Distribucion de goles totales por partido", 20, True),
        ],
        "boxplot": [
            # (df, columna, titulo, por)
            (df, "total_amarillas", "Tarjetas amarillas por partido en cada temporada", "temporada"),
            (df, "total_goles", "Goles por partido en cada temporada", "temporada"),
        ],
        "pastel": [
            # (valores, titulo, dona)
            (resultados, "Resultados finales de los partidos", False),
            (top_victorias, "Victorias como local: equipos con mas triunfos", True),
        ],
    }


def generar_grafica(tipo: str, params: tuple):
    """Selector multiple: elige la funcion de grafico segun el tipo y la ejecuta con la tupla de parametros."""
    match tipo:
        case "heatmap":
            return generar_heatmap(*params)
        case "scatter":
            return generar_scatterplot(*params)
        case "histograma":
            return generar_histplot(*params)
        case "boxplot":
            return generar_boxplot(*params)
        case "pastel":
            return generar_pastel(*params)
        case _:
            print(f"Tipo de grafica desconocido: {tipo}")
            return None


if __name__ == "__main__":
    df = read_dataset()
    if df is not None:
        df = preparar_datos(df)
        crear_carpeta(CARPETA_IMG)
        configuracion = construir_configuracion(df)
        for tipo, lista_params in configuracion.items():
            for i, params in enumerate(lista_params, start=1):
                figura = generar_grafica(tipo, params)
                if figura is not None:
                    ruta = guardar_figura(figura, os.path.join(CARPETA_IMG, f"{tipo}_{i}.png"))
                    print(f"Grafica guardada: {ruta}")
