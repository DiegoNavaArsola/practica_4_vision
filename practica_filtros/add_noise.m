
img_original = imread("Práctica 4\practica_filtros\eagle_snake.jpeg")
img_noise = imnoise(img_original,"gaussian")

imwrite(img_noise,"eagle_snake_ruido.png")