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
        return filtered_image
    else:
        return map_interpolate(filtered_image)

if __name__ == "__main__":

    """
    1) Imagen con filtro y sin filtro
    """
    # Cargar la imagen
    img = cv2.imread('mujer_dormida.jpg')
    #img_noise = cv2.imread("eagle_snake_ruido.png")

    # Convertir la imagen de BGR a RGB (para visualizarla correctamente con matplotlib)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("image_gray.png", img_gray)

    # Generar la imagen con ruido
    img_gray_noise = np.clip(img_gray + np.random.normal(0,25, img_gray.shape), 0, 255).astype(np.uint8)
    cv2.imwrite("image_noise.png", img_gray_noise)

    """
    2) Filtros paso bajas de bloque
    """
    # FIltros de bloque
    block_3x3 = calculate_block_filter(3, 3)
    block_7x7 = calculate_block_filter(7, 7)
    block_9x9 = calculate_block_filter(9, 9)
    block_11x11 = calculate_block_filter(11, 11)

    """
    3) Filtros paso bajas binomiales
    """
    # Filtros gaussianos
    gauss_3x3 = calculate_gauss_filter(3)
    gauss_7x7 = calculate_gauss_filter(7)
    gauss_9x9 = calculate_gauss_filter(9)
    gauss_11x11 = calculate_gauss_filter(11)

    """
    4) Filtros de borde (primera derivada)
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
    edge_11x11 = calculate_edge_filter(11)

    """
    5) Filtros laplacianos (segunda derivada)
    """
    # Laplaciano de literatura
    laplace_3x3_lit = (1 / 8) * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Segunda derivada
    laplace_5x5 = calculate_lapace_filter(5)
    laplace_7x7 = calculate_lapace_filter(7)
    laplace_11x11 = calculate_lapace_filter(11)

    """
    6) Difuminación y filtros unsharp-masking
    """
    # Suavizar imagen con filtro gauss 5x5
    img_smoth = apply_kernel(img_gray, calculate_gauss_filter(5),True)

    # Valor del peso para unsharp_masking
    k = 5

    # Obtención de los filtros
    unsharp_mask_block_3x3 = calculate_unsharp_masking(3, k, "block")
    unsharp_mask_block_7x7 = calculate_unsharp_masking(7, k, "block")

    unsharp_mask_gauss_3x3 = calculate_unsharp_masking(3, k, "gauss")
    unsharp_mask_gauss_7x7 = calculate_unsharp_masking(7, k, "gauss")


    """
    Obtención de las imagenes
    """
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
        "laplace_11x11": laplace_11x11,

        "smooth_image_5x5": calculate_gauss_filter(5),

        "unsharp_mask_block_3x3": unsharp_mask_block_3x3,
        "unsharp_mask_block_7x7": unsharp_mask_block_7x7,
        "unsharp_mask_gauss_3x3": unsharp_mask_gauss_3x3,
        "unsharp_mask_gauss_7x7": unsharp_mask_gauss_7x7
    }

    # Lista con los filtros que se normalizarán aun valor gris
    to_normalize = [
        "prewitt_x",
        "prewitt_y",
        "sobel_x",
        "sobel_y",
    ]

    #Aplicación de los filtros a cada imagen
    for kernel_name, kernel in filters_dict.items():
        # Aplicación de los filtros a la imagen sin ruido
        img_filtered = apply_kernel(img_gray, kernel, False)
        img_normalized_gray = apply_kernel(img_gray, kernel, True)

        # Aplicación de los filtros a la imagen con ruido
        img_filtered_noise = apply_kernel(img_gray_noise, kernel, False)
        img_normalized_gray_noise = apply_kernel(img_gray_noise, kernel, True)

        if kernel_name in to_normalize:
            cv2.imwrite(str(kernel_name + ".png"), img_normalized_gray)
            cv2.imwrite(str(kernel_name + "_noise.png"), img_normalized_gray_noise)
        else:
            cv2.imwrite(str(kernel_name + ".png"), img_filtered)
            cv2.imwrite(str(kernel_name + "_noise.png"), img_filtered_noise)