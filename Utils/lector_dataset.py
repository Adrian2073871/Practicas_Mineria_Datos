from pandas import DataFrame
import pandas as pd
import os
def read_dataset()->DataFrame:
    try:
        # Ruta relativa a este archivo, para no depender del directorio desde donde se ejecute
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_futbol_liga_espanola_limpio.csv")
        df = pd.read_csv(ruta)
        return df
    except FileNotFoundError:
        print("El archivo no se encuentra")
        return None

    