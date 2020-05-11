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

image_1 = np.asarray(Image.open("data/RubberWhale/frame10.png")).astype(float)
image_2 = np.asarray(Image.open("data/RubberWhale/frame11.png")).astype(float)

# GPU
matrice22 = np.array([[image_1,image_1],[image_1,image_1]])
matrice11 = np.array([image_1,image_1])
d_image_1 = cu.to_device(image_1.astype(float))
d_image_2 = cu.to_device(image_2.astype(float))
d_dx = cu.device_array_like(image_1.astype(float))
d_dy = cu.device_array_like(image_1.astype(float))
d_matrice22 = cu.to_device(matrice22.astype(float))
d_matrice11 = cu.to_device(matrice11.astype(float))
d_tab_dx = cu.device_array_like(image_1)


# BlockSize = np.array([32,32])
# gridSize = (np.asarray(image_1.shape) + (BlockSize-1))//BlockSize

    #----------------------------------------- Pour le Flux Optique normal -----------------------------------------#
# image_dy = op.derivee_y(image_1)
# image_dx = op.derivee_x(image_1)
# image_dt = op.derivee_t(image_1,image_2)
# plt.imshow(image_1)
# plt.imshow(image_dx)
# plt.imshow(image_dy)
# plt.imshow(image_dt)
# plt.colorbar()
# plt.show()



# image_dx, image_dy = op.flux_optique(image_1,image_2,29)
# flowpy.show_flow(image_dx,image_dy)

    #----------------------------------------- Pour le Flux Optique Vidéo -----------------------------------------#

# op.flux_optique_video("data/shibuya.mp4",21)
# op.show_camera()

    #------------------------------------------ Pour le Flux Optique GPU ------------------------------------------#

# 2 images
# image_dx, image_dy = op.flux_optique_GPU(d_image_1,d_image_2,21,d_matrice22,d_matrice11)
# flowpy.show_flow(image_dx,image_dy)

# video
op.flux_optique_video_GPU("data/shibuya.mp4",21)
    #----------------------------------------------- Pour des tests -----------------------------------------------#

# BlockSize = np.array([32,32])
# gridSize  = (np.asarray(image_1.shape) + (BlockSize-1))//BlockSize
# op.derivee_x_GPU[list(gridSize), list(BlockSize)](d_image_1, d_tab_dx)
# tab_dx = d_tab_dx.copy_to_host()
# plt.imshow(tab_dx)
# plt.colorbar()
# plt.show()
# open("data/shibuya.mp4")
# print(image_1)
