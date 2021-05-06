
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import argparse

def total_conv(imagen,filtro,verbose=False):
    imagen = addpadding(imagen,filtro,verbose)
    print(imagen)
    x=len(imagen)- len(filtro)+1
    y=len(imagen[0])- len(filtro)+1
    res = np.zeros((x,y))

    for i in range(x):
        for j in range(y):
            res[i][j] = np.sum(filtro * imagen[i: i+len(filtro), j: j+len(filtro[0])])
    plt.imshow(res, cmap='gray')
    plt.show()

    return res

def addpadding(imagen, kernel, verbose=False):
    if len(imagen.shape) == 3:
        print("Found 3 Channels : {}".format(imagen.shape))
        imagen = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(imagen.shape))
    else:
        print("Image Shape : {}".format(imagen.shape))


    x_kernel, y_kernel = kernel.shape

    pad_x = int((x_kernel-1)/2)
    pad_y = int((y_kernel-1)/2)
    
    x = len(imagen) + pad_x*2
    y = len(imagen[0]) + pad_y*2


    imagenPadding = np.zeros((x,y))
    for ren in range(pad_x, len(imagenPadding)-pad_x):
        for col in range(pad_y, len(imagenPadding[0])-pad_y):
            imagenPadding[ren,col] = imagen[ren - pad_x,col - pad_y]
            
    return imagenPadding



filtro = np.random.randint(10,size =(3,3))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])

print(total_conv(image,filtro,verbose=True))