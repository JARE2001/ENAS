
import numpy as np
import matplotlib.pyplot as plt
import random
#import cv2

 


def total_conv(padding,imagen,filtro):
    imagen = addpadding(padding,imagen)
    print(imagen)
    x=len(imagen[0])- len(filtro)+1
    y=len(imagen)- len(filtro)+1
    res = np.zeros((x,y))

    for i in range(x):
        for j in range(y):
            res[i][j] = np.sum(filtro * imagen[j:j+len(filtro[0]),i:i+len(filtro)])
    plt.imshow(res, cmap='gray')
    plt.show()

    return res

def addpadding(padding,imagen):
    x=len(imagen) + padding*2
    y=len(imagen[0]) + padding*2
    imagenPadding = np.zeros((x,y))
    for ren in range(padding, len(imagenPadding)-padding):
        for col in range(padding, len(imagenPadding[0])-padding):
            imagenPadding[ren,col] = imagen[ren - padding,col - padding]
    return imagenPadding


imgx= random.randint(4,6)
imgy=random.randint(4,6)

imagen =np.random.randint(10,size =(imgx,imgy))


# imagen = np.array( [[1, 2, 3, 4, 5, 6],
#                     [7, 8, 9, 10, 11, 12],
#                     [0, 0, 1, 16, 17, 18],
#                     [0, 1, 0, 7, 23, 24],
#                     [1, 7, 6, 5, 4, 3] ])

filtro = np.random.randint(10,size =(3,3))



print(total_conv(2,imagen,filtro))