"""
Tablas finales de posiciones de La Liga (2011-12 a 2025-26).

Uso:
    from tablas_liga import obtener_tablas
    tablas = obtener_tablas("resultados_futbol_liga_espanola_limpio.csv")
    tablas["2024-2025"]
"""
from functools import cmp_to_key
from Utils.lector_dataset import read_dataset
import pandas as pd

PARTIDOS_POR_TEMPORADA = 380
EQUIPOS_ESPERADOS = 20
COLUMNAS = ["Posicion", "Equipo", "PJ", "G", "E", "P", "GF", "GC", "DG", "PUNTOS"]

def cargar_partidos()->pd.DataFrame:
    df = read_dataset()
    df = df[["date", "home_team", "away_team",
             "home_goals_full_time", "away_goals_full_time"]].copy()
    df.columns = ["fecha", "local", "visitante", "gl", "gv"]
    return df



def dividir_temporadas(df: pd.DataFrame) -> dict:
    """Divide en bloques consecutivos de 380 partidos (el dataset viene en orden)."""
    if len(df) % PARTIDOS_POR_TEMPORADA != 0:
        raise ValueError(
            f"{len(df)} partidos no es múltiplo de {PARTIDOS_POR_TEMPORADA}."
        )
    temporadas = {}
    for i in range(len(df) // PARTIDOS_POR_TEMPORADA):
        bloque = df.iloc[i * PARTIDOS_POR_TEMPORADA:(i + 1) * PARTIDOS_POR_TEMPORADA]
        anio_inicio = pd.to_datetime(bloque["fecha"].iloc[0], format="%d/%m/%Y").year
        temporadas[f"{anio_inicio}-{anio_inicio + 1}"] = bloque
    return temporadas


# --------------------------------------------------------------------------- #
# 2. Estadísticas y desempates
# --------------------------------------------------------------------------- #
def _calcular_estadisticas(partidos: pd.DataFrame) -> pd.DataFrame:
    """Calcula PJ, G, E, P, GF, GC, DG y PUNTOS por equipo (3-1-0)."""
    local = pd.DataFrame({
        "Equipo": partidos["local"],
        "GF": partidos["gl"], "GC": partidos["gv"],
    })
    visita = pd.DataFrame({
        "Equipo": partidos["visitante"],
        "GF": partidos["gv"], "GC": partidos["gl"],
    })
    t = pd.concat([local, visita], ignore_index=True)
    t["PJ"] = 1
    t["G"] = (t["GF"] > t["GC"]).astype(int)
    t["E"] = (t["GF"] == t["GC"]).astype(int)
    t["P"] = (t["GF"] < t["GC"]).astype(int)

    tabla = t.groupby("Equipo", as_index=False)[["PJ", "G", "E", "P", "GF", "GC"]].sum()
    tabla["DG"] = tabla["GF"] - tabla["GC"]
    tabla["PUNTOS"] = tabla["G"] * 3 + tabla["E"]
    return tabla


def _desempate_duelo_directo(equipos: list, partidos: pd.DataFrame) -> list:
    """
    Ordena un grupo de equipos empatados en puntos, DG y GF según los
    partidos jugados entre ellos (duelo directo):
        1) puntos en los enfrentamientos directos
        2) diferencia de goles en los enfrentamientos directos
        3) goles a favor como visitante en los enfrentamientos directos
        4) orden alfabético (solo para que el resultado sea determinista)
    Con 2 equipos equivale a "el que ganó el duelo directo queda arriba".
    """
    directos = partidos[partidos["local"].isin(equipos) & partidos["visitante"].isin(equipos)]
    mini = {e: {"pts": 0, "dg": 0, "gv": 0} for e in equipos}

    for _, p in directos.iterrows():
        l, v, gl, gv = p["local"], p["visitante"], p["gl"], p["gv"]
        mini[l]["dg"] += gl - gv
        mini[v]["dg"] += gv - gl
        mini[v]["gv"] += gv
        if gl > gv:
            mini[l]["pts"] += 3
        elif gl < gv:
            mini[v]["pts"] += 3
        else:
            mini[l]["pts"] += 1
            mini[v]["pts"] += 1

    return sorted(
        equipos,
        key=lambda e: (-mini[e]["pts"], -mini[e]["dg"], -mini[e]["gv"], e),
    )


def _ordenar_tabla(tabla: pd.DataFrame, partidos: pd.DataFrame) -> pd.DataFrame:
    """Ordena por PUNTOS, DG, GF y, si persiste el empate, por duelo directo."""
    tabla = tabla.sort_values(
        ["PUNTOS", "DG", "GF"], ascending=[False, False, False], kind="mergesort"
    ).reset_index(drop=True)

    orden_final = []
    claves = tabla[["PUNTOS", "DG", "GF"]].apply(tuple, axis=1)
    i = 0
    while i < len(tabla):
        j = i
        while j + 1 < len(tabla) and claves[j + 1] == claves[i]:
            j += 1
        grupo = list(tabla["Equipo"].iloc[i:j + 1])
        if len(grupo) > 1:
            grupo = _desempate_duelo_directo(grupo, partidos)
        orden_final.extend(grupo)
        i = j + 1

    tabla = tabla.set_index("Equipo").loc[orden_final].reset_index()
    return tabla


def _nivel_rendimiento(posicion: int) -> str:
    if posicion <= 5:
        return "alto"
    if posicion <= 13:
        return "medio"
    return "bajo"


# --------------------------------------------------------------------------- #
# 3. Función principal
# --------------------------------------------------------------------------- #
def obtener_tablas(ruta_csv: str = "resultados_futbol_liga_espanola_limpio.csv") -> dict:
    """
    Retorna {"2011-2012": DataFrame, ..., "2025-2026": DataFrame}.
    Columnas: Posicion, Equipo, PJ, G, E, P, GF, GC, DG, PUNTOS, nivel_rendimiento.
    """
    partidos = cargar_partidos()
    resultado = {}

    for temporada, bloque in dividir_temporadas(partidos).items():
        tabla = _calcular_estadisticas(bloque)

        # Validaciones de integridad
        assert len(tabla) == EQUIPOS_ESPERADOS, f"{temporada}: {len(tabla)} equipos"
        assert (tabla["PJ"] == 38).all(), f"{temporada}: algún equipo no jugó 38 partidos"

        tabla = _ordenar_tabla(tabla, bloque)
        tabla["Posicion"] = range(1, len(tabla) + 1)
        tabla["nivel_rendimiento"] = tabla["Posicion"].apply(_nivel_rendimiento)
        resultado[temporada] = tabla[COLUMNAS + ["nivel_rendimiento"]]

    return resultado

