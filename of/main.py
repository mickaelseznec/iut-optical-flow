from PIL import Image
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import operators as op
import flowpy
import scipy.signal
import cv2 as cv
import numba.cuda as cu


    #------------------------------------------ Définition des variables -------------------------------------------#

image_1 = np.asarray(Image.open("data/RubberWhale/frame10.png"))
image_2 = np.asarray(Image.open("data/RubberWhale/frame11.png"))
# GPU
d_tab_dx = cu.device_array_like(image_1)
d_image_1 = cu.to_device(image_1)
d_image_2 = cu.to_device(image_2)




image_4 = np.array([4,5,7,1,0,6,3,3,2,8,4,1]).reshape(4,3)
d_image_4 = cu.to_device(image_4)
d_somme_tab = cu.device_array_like(d_image_4)
BlockSize = np.array([32,32])
gridSize = (np.asarray(image_4.shape) + (BlockSize-1))//BlockSize

    #----------------------------------------- Pour le Flux Optique normal -----------------------------------------#
# image_dy = op.derivee_y(image_1)
# image_dx = op.derivee_x(image_1)
# image_dt = op.derivee_t(image_1,image_2)
# plt.imshow(image_1)
# plt.imshow(image_dx)
# plt.imshow(image_dy)
# plt.imshow(image_dt)
# plt.show()



# image_dx, image_dy = op.flux_optique(image_1,image_2,29)
# flowpy.show_flow(image_dx,image_dy)

    #----------------------------------------- Pour le Flux Optique Vidéo -----------------------------------------#

op.flux_optique_video("data/shibuya.mp4",21)
# op.show_camera()

    #------------------------------------------ Pour le Flux Optique GPU ------------------------------------------#

# op.somme_fenetre_global_GPU[list(gridSize), list(BlockSize)](d_image_4, 1, d_somme_tab)
# somme_GPU = d_somme_tab.copy_to_host()
# print(somme_GPU)
    #----------------------------------------------- Pour des tests -----------------------------------------------#

# BlockSize = np.array([32,32])
# gridSize  = (np.asarray(image_1.shape) + (BlockSize-1))//BlockSize
# d_tab_dx = op.derivee_x_GPU[list(gridSize), list(BlockSize)](d_image_1, d_tab_dx)
# open("data/shibuya.mp4")
# print(image_1)
# 