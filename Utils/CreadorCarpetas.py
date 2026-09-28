import os
import errno
def crear_carpeta(path:str)->True:
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("El directorio ya existe.")
            raise
    except AttributeError:
        print("No se pudo crear el directorio porque no se le asigno un nombre.")
    else:
        return True
    return False