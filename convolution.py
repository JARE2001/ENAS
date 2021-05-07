import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import argparse

# Funcion de convulsion de una imagen con un filtro dado
def total_conv(imagen,filtro,nombre,verbose=False):
    imagen = addpadding(imagen,filtro,verbose) # Se le agrega padding a la imagen
    print(imagen)
    x=len(imagen)- len(filtro)+1
    y=len(imagen[0])- len(filtro)+1
    res = np.zeros((x,y))

    # Operacion para la convulsion
    for i in range(x):
        for j in range(y):
            res[i][j] = np.sum(filtro * imagen[i: i+len(filtro), j: j+len(filtro[0])])

    # Grafica la imagen con el filtro
    plt.imshow(res, cmap='gray')
    plt.title("Output " +nombre+ " Convolution")
    plt.show()

    gradient_magnitude = np.sqrt(np.square(res))

    # Grafica la imagen con el filtro añadido y magnitud gradiente
    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude ("+nombre+")")
        plt.show()

    return res

# Agrega padding a la imagen
def addpadding(imagen, kernel, verbose=False):
    # Si es una imagen RGB la convierte a una escala de grises
    if len(imagen.shape) == 3:
        print("Found 3 Channels : {}".format(imagen.shape))
        imagen = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(imagen.shape))
    else:
        print("Image Shape : {}".format(imagen.shape))

    x_kernel, y_kernel = kernel.shape

    # Se calcula el tamaño del padding de la imagen
    pad_x = int((x_kernel-1)/2)
    pad_y = int((y_kernel-1)/2)
    
    # Define el tamaño de la matriz que tendrá la imagen con padding
    x = len(imagen) + pad_x*2
    y = len(imagen[0]) + pad_y*2

    imagenPadding = np.zeros((x,y)) # Se crea una matriz del tamaño de la imagen con padding

    # Se añade la imagen a la matriz con padding añadido
    for ren in range(pad_x, len(imagenPadding)-pad_x):
        for col in range(pad_y, len(imagenPadding[0])-pad_y):
            imagenPadding[ren,col] = imagen[ren - pad_x,col - pad_y]

    # Grafica la imagen con padding
    if verbose:
        plt.imshow(imagenPadding, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    return imagenPadding

# Filtro para agregar un filtro de Laplacian
filtro_laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

# Filtro para agrefar un filtro de Sobel
filtro_sobel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

# Filtro para agrefar un filtro de Gaussian Blur
filtro_gaussian = np.array([[0,0,0,5,0,0,0],[0,5,18,32,18,5,0],[0,18,64,100,64,18,0],[5,32,100,100,100,32,5],[0,18,64,100,64,18,0],[0,5,18,32,18,5,0],[0,0,0,5,0,0,0]])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])

    # Convulsion de la imagen con cada filtro
    imagenLaplacian = total_conv(image,filtro_laplacian,"Laplacian",verbose=True)
    imagenSobelH =total_conv(image,filtro_sobel,"Sobel Horizontal",verbose=True)
    imagenSobelV =total_conv(image,np.flip(filtro_sobel.T, axis=0),"Sobel Vertical",verbose=True)
    imagenGaussianBlur = total_conv(image, filtro_gaussian,"Gaussian Blur",verbose=True)

    gradient_magnitude = np.sqrt(np.square(imagenLaplacian)+np.square(imagenSobelH)+np.square(imagenSobelV))
    #+np.square(imagenGaussianBlur)

    # Grafica la imagen con todos los fitros combinados y magnitud gradiente
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title("Gradient Magnitude Total")
    plt.show()