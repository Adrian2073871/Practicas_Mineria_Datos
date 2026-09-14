import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import metricas_datos_agrupados
from metricas_datos_agrupados import crear_metricas_datos_agrupados
#Calcular medidas de tendencia central en Series
def calcular_media(valores):
    return  valores.mean()

def calcular_mediana(valores):
    return valores.median()


#Calcular medidas de dispersion en Series
def calcular_varianza(valores):
    return valores.var(ddof=1)

def calcular_desv_estandar(valores):
    return valores.std(ddof=1)

#Calcular Correlacion entre 2 variables
#Recibe como parametros los objetos de tipo 'Series' que representan las variables a verificar su correlacion
def calcular_correlacion(x, y):
    return x.corr(y, method = 'pearson')

#Esta funcion crea un grafico 
def graficar_correlacion(x, y):
    plt.scatter(x,y)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title("Correlacion")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("resultados_futbol_liga_espanola_limpio.csv")
    print(df.info())
    print("#################Promedios de los tiros a puerta de equipos locales y visitantes#################")
    print(f"------------>Promedio de tiros a puerta de equipos locales:{calcular_media(df["home_shots_on_target"]):.2f}")
    print(f"------------>Promedio de tiros a puerta de equipos visitantes:{calcular_media(df["away_shots_on_target"]):.2f}")
    print("\n\n")
    print("#################Mediana de los tiros a puerta de equipos locales y visitantes#################")
    print(f"------------>Mediana de tiros a puerta de equipos locales:{calcular_mediana(df['home_shots_on_target']):.2f}")
    print(f"------------>Mediana de tiros a puerta de equipos visitantes:{calcular_mediana(df['away_shots_on_target']):.2f}")
    print("\n\n")
    print(f"------------>Promedio de faltas cometidas por el equipo visitante:{calcular_media(df["away_fouls"]):.2f}")
    print(f"------------>Desviacion estandar de faltas cometidas por el equipo visitante:{calcular_desv_estandar(df["away_fouls"]):.2f}")
    print("\n\n")
    print(f"""Coeficiente de correlacion entre:\n
    X->tiros a puerta de los equipos locales.\n
    Y->goles anotados de los equipos locales.\n
    Coeficiente de Pearson: {calcular_correlacion(df['home_shots_on_target'], df['home_goals_full_time'])}
""")
    graficar_correlacion(df['home_shots_on_target'], df['home_goals_full_time'])

    print(f"""Coeficiente de correlacion entre:\n
        X->tiros a puerta de los equipos visitantes.\n
        Y->goles anotados de los equipos visitantes.\n
        Coeficiente de Pearson: {calcular_correlacion(df['away_shots_on_target'], df['away_goals_full_time'])}
    """)
    graficar_correlacion(df['away_shots_on_target'], df['away_goals_full_time'])

    #Obtener metricas para datos agrupados
    """
    Se agruparan los datos en base a la columna de equipos locales y se utilizara la cantidad de goles hechos por partido de locales.
    """
    metricas = list(["sum", "mean", "max", "std"])
    gdf = crear_metricas_datos_agrupados("home_team", "home_goals_full_time", df, metricas)
    print(gdf)