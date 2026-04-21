import os

# La ruta que pusiste en tu script
RUTA = "lfw-deepfunneled/" 

print(f"Buscando en: {os.path.abspath(RUTA)}")

if not os.path.exists(RUTA):
    print("¡ERROR! La carpeta no existe en esa ruta.")
else:
    contenido = os.listdir(RUTA)
    print(f"Contenido encontrado ({len(contenido)} items):")
    print(contenido[:5]) # Muestra los primeros 5