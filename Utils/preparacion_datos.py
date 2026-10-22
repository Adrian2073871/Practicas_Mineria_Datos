import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def estandarizar_datos(df:DataFrame)->dict:
    sc = StandardScaler()
    df_scaled = sc.fit_transform(df)
    normalizacion = dict()
    normalizacion["escalador"] = sc
    normalizacion["df_scaled"] = df_scaled
    return normalizacion

def dividir_datos_prueba_entrenamiento(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23,
                                                        stratify=y)
    return x_train, x_test, y_train, y_test