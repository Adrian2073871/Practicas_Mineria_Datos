import subprocess
import os
def limpiar_consola():
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "cls"],
            check=True
        )
    else:
        subprocess.run(
            ["clear"],
            check=True
        )
  

def pausar_programa():
    input("\nPresione Enter para continuar....")