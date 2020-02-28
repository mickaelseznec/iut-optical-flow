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
d_tab_dx = cu.device_array_like(image_1)
d_image_1 = cu.to_device(image_1)

    #----------------------------------------- Pour le Flux Optique normal -----------------------------------------#

image_dx, image_dy = op.flux_optique(image_1,image_2,29)
flowpy.show_flow(image_dx,image_dy)

    #----------------------------------------- Pour le Flux Optique Vidéo -----------------------------------------#

# op.flux_optique_video("data/foot.mp4",21)
# op.show_camera()

    #----------------------------------------------- Pour des tests -----------------------------------------------#

# BlockSize = np.array([32,32])
# gridSize  = (np.asarray(image_1.shape) + (BlockSize-1))//BlockSize
# d_tab_dx = op.derivee_x_GPU[list(gridSize), list(BlockSize)](d_image_1, d_tab_dx)
# open("data/shibuya.mp4")
# print(image_1)
# image_dy = op.derivee_y(image_1)
# plt.subplot(1,2,1)
# plt.imshow(image_dx)
# plt.subplot(1,2,2)
# plt.imshow(image_dy)
# plt.show()