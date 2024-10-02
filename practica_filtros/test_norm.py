import cv2
import numpy as np

# Supongamos que 'img' es la imagen cargada que contiene valores negativos, positivos y ceros
# Ejemplo: puedes cargar una imagen con float32 para tener rango de valores positivos y negativos
# img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Crear una imagen de ejemplo con valores negativos y positivos
img = np.array([[-100, -50, 0, 50, 100],
                [-200, -100, 0, 100, 200],
                [-255, 0, 0, 0, 255]], dtype=np.float32)

# Encontrar el valor mínimo y máximo de la imagen
min_val = np.min(img)
max_val = np.max(img)

# Normalizar la imagen para que los valores estén en el rango [0, 255]
# Queremos que 0 se mapee a 127 (gris), valores negativos a 0 (negro) y positivos a 255 (blanco)
# Para ello, primero sumamos 127 a todos los valores
norm_img = np.clip((img - min_val) * 255 / (max_val - min_val), 0, 255)

# Convertir la imagen a tipo uint8
norm_img = norm_img.astype(np.uint8)

# Mostrar la imagen resultante
cv2.imshow('Imagen Normalizada', norm_img)
cv2.waitKey(0)
cv2.destroyAllWindows()