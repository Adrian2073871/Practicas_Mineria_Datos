from Utils.preparacion_datos import estandarizar_datos
import pandas as pd
from pandas import DataFrame
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def reducir_dimensionalidad(n:int, df_general:DataFrame, X_features: DataFrame, columna_clases:str)->Figure:
    #Escalar las caracteristicas numericas del dataset
    estandarizador = estandarizar_datos(X_features)
    sc = estandarizador["escalador"]
    x_scaled = estandarizador["df_scaled"]
    #Reducir a 2 dimensiones
    pca = PCA(n_components=n)
    x_reduce = pca.fit_transform(x_scaled)
    #Crear el diagrama de dispersion

    fig = plt.figure(figsize=(10, 7))
    for clase in df_general[columna_clases].unique():
        mascara = df_general[columna_clases] == clase
        plt.scatter(
            x_reduce[mascara.to_numpy(), 0],
            x_reduce[mascara.to_numpy(), 1],
            label=clase,
            alpha=0.7
        )
        plt.xlabel("Componente principal 1")
        plt.ylabel("Componente principal 2")
        plt.title("Equipos por nivel de rendimiento (PCA)")
        plt.legend(title="Nivel de rendimiento")
        plt.grid(alpha=0.3)
        plt.tight_layout()
    print("Varianza explicada por componente:", pca.explained_variance_ratio_)
    print("Varianza explicada acumulada:", pca.explained_variance_ratio_.sum())
    return fig



