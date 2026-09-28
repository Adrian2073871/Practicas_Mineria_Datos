import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import Series
from matplotlib.figure import Figure

sns.set_theme(style="whitegrid")


def guardar_figura(fig: Figure, ruta: str) -> str:
    """Guarda la figura como imagen en 'ruta' y la cierra para liberar memoria."""
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def generar_heatmap(df: DataFrame, titulo: str, filas: list = None, columnas: list = None) -> Figure | None:
    """
    Mapa de calor de la matriz de correlacion de las variables int64.
    Si se indican 'filas' y 'columnas' se muestra solo ese bloque de la matriz
    (util para cruzar, por ejemplo, variables del local contra las del visitante).
    """
    data = df.select_dtypes(include="int64")
    if data.empty:
        print("No se encontraron columnas con datos numericos")
        return None
    # Generar la matriz de correlacion
    matriz_corr = data.corr()
    cuadrado = True
    if filas is not None and columnas is not None:
        matriz_corr = matriz_corr.loc[filas, columnas]
        cuadrado = False
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=cuadrado,
        linewidths=0.5
    )
    plt.title(titulo)
    plt.xticks(rotation=45, ha="right")
    return fig


def generar_scatterplot(X: str, Y: str, df: DataFrame, titulo: str,
                        hue: str = None, etiqueta: str = None, linea: bool = False) -> Figure:
    """
    Diagrama de dispersion entre las columnas X e Y de df.
    """
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x=X,
        y=Y,
        hue=hue,
        alpha=0.5 if etiqueta is None else 0.9
    )
    if linea:
        sns.regplot(data=df, x=X, y=Y, scatter=False, ci=None, color="gray",
                    line_kws={"linestyle": "--"})
    if etiqueta is not None:
        for _, fila in df.iterrows():
            plt.annotate(fila[etiqueta], (fila[X], fila[Y]), fontsize=8,
                         xytext=(4, 4), textcoords="offset points")
    plt.title(titulo)
    return fig


def generar_histplot(column: str, df: DataFrame, titulo: str, bins: int = 20,
                     discrete: bool = False) -> Figure:
    """Histograma con curva KDE. Con discrete=True cada valor entero tiene su propia barra (sin KDE)."""
    fig = plt.figure(figsize=(12, 10))
    if discrete:
        sns.histplot(data=df, x=column, discrete=True, kde=False)
    else:
        sns.histplot(data=df, x=column, bins=bins, kde=True)
    plt.title(titulo)
    plt.ylabel("Frecuencia")
    return fig


def generar_boxplot(df: DataFrame, column: str, titulo: str, por: str = None) -> Figure:
    """Diagrama de bigote de 'column'. Si se indica 'por', genera una caja por cada categoria de esa columna."""
    fig = plt.figure(figsize=(12, 10))
    sns.boxplot(
        data=df,
        x=column,
        y=por,
        orient="h"
    )
    plt.title(titulo)
    return fig


def generar_pastel(valores: Series, titulo: str, dona: bool = False) -> Figure:
    """
    Grafico circular. 'valores' es una Series cuyo indice son las etiquetas
    y cuyos valores son las cantidades a repartir. dona=True lo dibuja como dona.
    """
    fig = plt.figure(figsize=(9, 9))
    wedgeprops = {"width": 0.45, "edgecolor": "white"} if dona else {"edgecolor": "white"}
    plt.pie(
        valores,
        labels=valores.index,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=sns.color_palette("pastel", len(valores)),
        wedgeprops=wedgeprops,
        pctdistance=0.78 if dona else 0.6
    )
    plt.title(titulo)
    plt.axis("equal")
    return fig
