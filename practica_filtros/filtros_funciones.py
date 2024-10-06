import numpy as np

def pascal_triangle(n):
    """
    :param n: Nivel del triángulo
    :return: Lista con los coeficientes presentes en el nivel n del triángulo de pascal
    """

    triangle = []
    if n <= 0:
        return [1]
    else:
        for i in range(n):
                row = [None for element in range(i + 1)]
                row[0], row[-1] = 1, 1
                for j in range(1, i):
                    row[j] = triangle[i-1][j-1] + triangle[i-1][j]
                triangle.append(row)
        return triangle[-1]

def first_derivate(n):
    """
    :param n: Nivel del triángulo
    :return: Lista los coeficientes presentes en el triángulo derivado de pascal en el nivel n
    """

    inv_triangle = []
    if n > 0:
        row_pascal = pascal_triangle(n)

        inverse_row_pascal = [-1 * element for element in row_pascal]
        inverse_row_pascal.insert(0,0)

        row_pascal.append(0)

        for i in range(len(row_pascal)):
            inv_triangle.append(row_pascal[i] + inverse_row_pascal[i])

        return inv_triangle

    else:
        return [1]

def second_derivate(n):
    """
    :param n: Nivel del triángulo
    :return: Lista los coeficientes presentes en el triángulo derivado de pascal en el nivel n
    """

    inv_triangle = []
    if n > 0:
        row_pascal = first_derivate(n)

        inverse_row_pascal = [-1 * element for element in row_pascal]
        inverse_row_pascal.insert(0,0)

        row_pascal.append(0)

        for i in range(len(row_pascal)):
            inv_triangle.append(row_pascal[i] + inverse_row_pascal[i])

        return inv_triangle

    else:
        return [1]

def sum_elements_matrix(matrix):
    """
    :param matrix: Matriz cualquiera
    :return: La suma de todos los elementos en la matriz
    """
    return np.sum(np.array(matrix))

def max_value_matrix(matrix):
    """
    :param matrix: Matriz cualquiera
    :return: El valor máximo de los elementos de la matriz
    """
    return np.max(np.abs(np.array(matrix)))

def gradient(n):
    """
    :param n:  Nivel del triángulo de pascal
    :return: La magnitud (matriz) del gradiente de la función matricial para n coeficientes
    """
    bin_fun = np.array([pascal_triangle(n)])
    sec_der = np.array([second_derivate(n-2)])
    return np.multiply(bin_fun.transpose(),sec_der) + np.multiply(sec_der.transpose(),bin_fun)

def identity_filter(n):
    # Crear una matriz de ceros de tamaño n x n
    identity = np.zeros((n, n))

    # Ubicar el 1 en el centro
    mid = n // 2
    identity[mid, mid] = 1

    return identity

def unite_separate_filters(array):
    """
    :param array: Matriz (array) de tamaño nx1
    :return: El filtro (kernel) nxn a partir del array y su transpuesta
    """
    return np.multiply(np.array(array).transpose(), np.array(array))

def calculate_gauss_filter(n):
    """
    :param n:  Nivel del triángulo de pascal
    :return: Kernel normalizado de tamaño nxn calculado a partir de los coeficientes del triángulo de pascal en el nivel n
             La normalización se obtiene al dividir los elementos del kernel entre las suma de todos estos
    """
    gauss_fun = pascal_triangle(n)
    united_matrix = unite_separate_filters([gauss_fun])
    sum = sum_elements_matrix(united_matrix)
    return (1 / sum) * united_matrix

def calculate_edge_filter(n):
    """
    :param n:  Nivel del triángulo de pascal
    :return: Kernel normalizado de tamaño nxn calculado a partir de los coeficientes del triángulo de pascal derivado en el nivel n
             La normalización se obtiene al dividir los elementos del kernel entre las valor más alto de estos
    """
    first_der = first_derivate(n)
    united_matrix = unite_separate_filters([first_der])
    max = max_value_matrix(united_matrix)
    return (1 / max) * united_matrix

def calculate_lapace_filter(n):
    """
    :param n:  Nivel del triángulo de pascal
    :return: Kernel normalizado de tamaño nxn calculado a partir de los coeficientes del triángulo de pascal derivado en el nivel n
             La normalización se obtiene al dividir los elementos del kernel entre las valor más alto de estos
    """
    grad = gradient(n)
    max = max_value_matrix(grad)
    return (1/max) * grad

def calculate_block_filter(n,m):
    return (1/(n*m))*np.ones((n,m), np.float32)

def calculate_prewitt(n,dir):
    dir = str(dir)
    filter = np.ones((n,1), np.float32) * first_derivate(n-1)
    filter_norm = (1/max_value_matrix(filter)) * filter

    if dir.lower() == "x":
        return filter_norm
    elif dir.lower() == "y":
        return filter_norm.transpose()
    else:
        print("Ingrese una dirección correcta (X o Y)")

def calculate_sobel(n,dir):
    dir = str(dir)
    gauss_fun = pascal_triangle(n)

    filter = np.array([gauss_fun]).transpose() * (-1) * first_derivate(n-1)
    filter_norm = (1/max_value_matrix(filter))  * filter


    if dir.lower() == "x":
        return filter_norm
    elif dir.lower() == "y":
        return filter_norm.transpose()
    else:
        print("Ingrese una dirección correcta (X o Y)")


def calculate_unsharp_masking(n, k, type):
    i = identity_filter(n)
    if  type.lower() == "block":
        block = calculate_block_filter(n,n)
        return i + k * (i - block)
    elif type.lower() == "gauss":
        gauss = calculate_gauss_filter(n)
        return i + k * (i - gauss)
    else:
        print("Tipo de filtro no válido")


if __name__ == "__main__":

    # Numero de coeficientes en el filtro
    num_coef = 5
    n = num_coef - 1



    """
    Cálculo de los triángulos de coeficientes
    """
    # Cálculo de la función gaussiana para n (triángulo de Pascal)
    t = pascal_triangle(num_coef)
    print(f"Funcion gaussiana con {num_coef} coeficientes (n={n}): ")
    print(t)
    print("\n")

    # Cálculo de la primera derivada para n (triángulo de Pascal diferenciado)
    d1_t = first_derivate(n)
    print(f"Primera derivada con {num_coef} coeficientes (n={n}): ")
    print(d1_t)
    print("\n")

    # Cálculo de la segunda derivada para n (triángulo de Pascal diferenciado)
    d2_t = second_derivate(n-1)
    print(f"Segunda derivada con {num_coef} coeficientes (n={n}): ")
    print(d2_t)
    print("\n")


    """
    Filtros sin normalizar
    """
    # Creación del filtro gaussiano o de suavizado (sin normalizar) a partir del vector
    united_gauss_filter = unite_separate_filters([t])
    print(f"Filtro gaussiano (sin normalizar) con {num_coef} coeficientes (n={n}): ")
    print(united_gauss_filter)
    print("\n")

    # Creación del filtro binomial o de de borde (sin normalizar) a partir del vector
    united_1st_der_filter = unite_separate_filters([d1_t])
    print(f"Filtro 1 derivada (sin normalizar) con {num_coef} coeficientes n={n}: ")
    print(united_1st_der_filter)
    print("\n")

    # Creación del filtro binomial o de de borde (sin normalizar) a partir del vector
    united_2nd_der_filter = gradient(num_coef)
    print(f"Filtro 2 derivada (sin normalizar) con {num_coef} coeficientes n={n}: ")
    print(united_2nd_der_filter)
    print("\n")

    """
    Filtros normalizados
    """
    # Normalizando la matriz con el promedio ponderado (filtro final)
    sum_ele_matrix = sum_elements_matrix(united_gauss_filter)
    f_normalized = calculate_gauss_filter(num_coef)
    print(f"Filtro gaussiano normalizado con {num_coef} coeficientes (n={n}) (divido entre {sum_ele_matrix}): ")
    print(f_normalized)
    print("\n")

    # Normalizando la matriz con el valor máximo presente dentro del filtro (filtro final)
    max_ele_1der = max_value_matrix(united_1st_der_filter)
    d1_t_normalized = calculate_edge_filter(n)
    print(f"Filtro de bordes (1 der) normalizado con {num_coef} coeficientes (n={n}) (divido entre {max_ele_1der}): ")
    print(d1_t_normalized)
    print("\n")

    # Normalizando la matriz con el valor máximo presente dentro del filtro (filtro final)
    max_ele_2der = max_value_matrix(united_2nd_der_filter)
    d2_t_normalized = calculate_lapace_filter(num_coef)
    print(f"Filtro laplaciano normalizado con {num_coef} coeficientes (n={n}) (divido entre {max_ele_2der}): ")
    print(d2_t_normalized)
    print("\n")