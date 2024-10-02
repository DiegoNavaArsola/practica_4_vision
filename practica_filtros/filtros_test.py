import cv2
import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
from filtros_funciones import *
from scipy.signal import convolve2d


def map_interpolate(image):
    h = np.size(image, 0)
    w = np.size(image, 1)
    new_img = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            new_img[i][j] = (float(image[i][j]) + 255) // 2

    return new_img

def apply_kernel(image, kernel, normalize):
    filtered_image = convolve2d(image, kernel, mode='same')
    if not normalize:
        #print(filtered_image)
        return filtered_image
    else:
        #print(map_interpolate(filtered_image))
        return map_interpolate(filtered_image)


# Cargar la imagen
imagen = cv2.imread('arte.png')

# Convertir la imagen de BGR a RGB (para visualizarla correctamente con matplotlib)
imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

"""
2)
"""
# FIltros de bloque
block_3x3 = calculate_block_filter(3, 3)
block_7x7 = calculate_block_filter(7, 7)
block_9x9 = calculate_block_filter(9, 9)
block_11x11 = calculate_block_filter(11, 11)

"""
3)
"""
# Filtros gaussianos
gauss_3x3 = calculate_gauss_filter(3)
gauss_7x7 = calculate_gauss_filter(7)
gauss_9x9 = calculate_gauss_filter(9)
gauss_11x11 = calculate_gauss_filter(11)

"""
4)
"""
# Filtro de bloque [1,-1]
block_1x2 = [np.array([1, -1])]

# Prewitt X y Y
prewitt_x = calculate_prewitt(5, "X")
prewitt_y = calculate_prewitt(5, "Y")

# Sobel X y Y
sobel_x = calculate_sobel(5, "X")
sobel_y = calculate_sobel(5, "Y")

# Primera derivada
edge_5x5 = calculate_edge_filter(5)
edge_7x7 = calculate_edge_filter(7)
edge_11x11 = calculate_edge_filter(1)

"""
5)
"""
# Laplaciano de literatura
laplace_3x3_lit = (1 / 8) * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Segunda derivada
laplace_5x5 = calculate_lapace_filter(5)
laplace_7x7 = calculate_lapace_filter(7)
laplace_11x11 = calculate_lapace_filter(11)

"""
6) 
"""

# Aplicar el filtro a la imagen

filters_dict = {

    "block_3x3": block_3x3,
    "block_7x7": block_7x7,
    "block_9x9": block_9x9,
    "block_11x11": block_11x11,

    "gauss_3x3": gauss_3x3,
    "gauss_7x7": gauss_7x7,
    "gauss_9x9": gauss_9x9,
    "gauss_11x11": gauss_11x11,

    "block_1x2": block_1x2,

    "prewitt_x": prewitt_x,
    "prewitt_y": prewitt_y,

    "sobel_x": sobel_x,
    "sobel_y": sobel_y,

    "edge_5x5": edge_5x5,
    "edge_7x7": edge_7x7,
    "edge_11x11": edge_11x11,

    "laplace_3x3_lit": laplace_3x3_lit,

    "laplace_5x5": laplace_5x5,
    "laplace_7x7": laplace_7x7,
    "laplace_11x11": laplace_11x11
}

for kernel_name, kernel in filters_dict.items():
    #img_filtered = cv2.filter2D(imagen_gray, -1, kernel)
    img_filtered = apply_kernel(imagen_gray, kernel, False)
    img_normalized_gray = apply_kernel(imagen_gray, kernel, True)
    if kernel_name == "prewitt_x" or kernel_name == "prewitt_y" or kernel_name == "sobel_x" or kernel_name == "sobel_y":
        cv2.imwrite(str(kernel_name + ".png"), img_normalized_gray)
    else:
        cv2.imwrite(str(kernel_name + ".png"), img_filtered)

img_filtered = apply_kernel(imagen_gray, gauss_11x11, False)
img_normalized = apply_kernel(imagen_gray, gauss_11x11, True)


# Mostrar la primera imagen en la primera posición (1, 3, 1)
plt.subplot(1, 2, 1)
plt.imshow(imagen_gray, cmap='gray')
plt.title('Imagen 1')
plt.axis('off')  # Para quitar los ejes

# Mostrar la segunda imagen en la segunda posición (1, 3, 2)
plt.subplot(1, 2, 2)
plt.imshow(img_normalized, cmap='gray')
plt.title('Imagen 2')
plt.axis('off')

# Mostrar todas las imágenes en una ventana
plt.show()
