#Este modulo contiene funciones para mostrar reportes de metricas para los diferentes algoritmos
#Reporte de metricas para algoritmos de clasificacion
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from pandas import(
    Series,
    DataFrame
)

def generar_matriz_confusion(y_test:Series, y_pred:Series)->DataFrame:
    matriz = confusion_matrix(y_test, y_pred)
    df_matriz = pd.DataFrame(matriz)
    return df_matriz
def generar_reporte_metricas_clasificacion(y_test:Series, y_pred:Series)->None:
    print(classification_report(
        y_test,
        y_pred,
        digits = 4,
        zero_division=0
    ))
    matriz_confusion = generar_matriz_confusion(y_test, y_pred)
    print("\nMatriz de confusion")
    print(matriz_confusion)
    print("\n========================================================\n")





