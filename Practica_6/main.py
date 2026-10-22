import pandas as pd
from Utils.lector_dataset import read_dataset
from Utils.tablas_liga import obtener_tablas
from Utils.CreadorCarpetas import crear_carpeta
from sklearn.neighbors import KNeighborsClassifier
from Utils.mostrar_resultados import generar_reporte_metricas_clasificacion
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from Utils.funciones_sistema import(
    limpiar_consola,
    pausar_programa
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Utils.preparacion_datos import(
    dividir_datos_prueba_entrenamiento,
    estandarizar_datos
)
from Practica_3.Graficas.graficas import(
    generar_heatmap,
    guardar_figura
)
import os
CARPETA_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
if __name__ == "__main__":
    tablas = obtener_tablas()
    crear_carpeta(CARPETA_IMG)
    #Tablas finales de cada temporada
    for temporada, df in tablas.items():
        print(f"\n===== Temporada {temporada} =====")
        print(df.to_string(index=False))
        fig = generar_heatmap(df, f"Relacion entre estadisticas de equipos_{temporada}")
        ruta = guardar_figura(fig, os.path.join(CARPETA_IMG, f"Correlacion de variables en la {temporada}_.png"))
        print(f"Grafica guardada en la ruta: {ruta}")

    #Tabla con todos los registros de todas las temporadas
    df_general = pd.concat(
        tablas.values(),
        ignore_index=True
    )
    """
    print(df_general.shape)
    print(df_general.head())
    """
    fig = generar_heatmap(df_general, f"Relacion entre estadisticas de equipos")
    ruta = guardar_figura(fig, CARPETA_IMG)
    print(f"Grafica guardada en la ruta: {ruta}")
    #Preparacion de caracteristicas
    X_features = df_general.drop(columns=['Posicion', 'Equipo', 'PJ', 'nivel_rendimiento', 'PUNTOS'])
    y = df_general["nivel_rendimiento"]
    #Dividir los datos en prueba y entrenamiento
    x_train, x_test, y_train, y_test = dividir_datos_prueba_entrenamiento(X_features, y)
    #Codificacion de las clases como numeros
    codificador = LabelEncoder()
    y_train_encode = codificador.fit_transform(y_train)
    y_test_encode = codificador.transform(y_test)
    #Estandarizacion de las caracteristicas
    objeto_estandarizador = estandarizar_datos(x_train)
    sc = objeto_estandarizador["escalador"]
    x_train_scaled = objeto_estandarizador["df_scaled"]
    #Transformando los datos de prueba con el mismo escalador
    x_test_scaled = sc.transform(x_test)
    #Reducir a 2 dimensiones las caracteristicas
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    #Valores de k para utilizar
    k_values = [3,5,7,9,11,13,15]
    for k in k_values:
        print(f"\n=====Para k = {k} =====")
        #Entrenar el modelo de KNN
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(x_train_pca, y_train_encode)
        #Evaluar el modelo
        y_pred = pd.Series(modelo.predict(x_test_pca))
        #Generacion de reporte de evaluacion de rendimiento
        generar_reporte_metricas_clasificacion(y_test_encode, y_pred)
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_decision_regions(
            X=x_train_pca,
            y=y_train_encode,
            clf=modelo,
            legend=2,
            ax=ax
        )
        # Cambiar los números por los nombres originales
        handles, _ = ax.get_legend_handles_labels()

        ax.legend(
            handles,
            codificador.classes_,
            title="Resultado del partido"
        )
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.set_title(f"Fronteras de decision KNN (k={k})")
        fig.tight_layout()
        ruta = guardar_figura(fig, os.path.join(CARPETA_IMG, f"KNN with {k}_.png"))
        print(f"Grafica guardada en la ruta: {ruta}")
        pausar_programa()
        limpiar_consola()
    print("Varianza explicada por componente:", pca.explained_variance_ratio_)
    print("Varianza explicada acumulada:", pca.explained_variance_ratio_.sum())












