
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import argparse

def total_conv(padding,imagen,filtro):
    imagen = addpadding(padding,imagen)
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

def addpadding(padding,imagen):

    if len(imagen.shape) == 3:
        print("Found 3 Channels : {}".format(imagen.shape))
        imagen = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(imagen.shape))
    else:
        print("Image Shape : {}".format(imagen.shape))

    x=len(imagen) + padding*2
    y=len(imagen[0]) + padding*2
    imagenPadding = np.zeros((x,y))
    for ren in range(padding, len(imagenPadding)-padding):
        for col in range(padding, len(imagenPadding[0])-padding):
            imagenPadding[ren,col] = imagen[ren - padding,col - padding]
    return imagenPadding


filtro = np.random.randint(10,size =(3,3))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])

print(total_conv(2,image,filtro))