from PIL import Image
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import operators as op
import flowpy
import scipy.signal

image_1 = np.asarray(Image.open("data/RubberWhale/frame10.png"))
image_2 = np.asarray(Image.open("data/RubberWhale/frame11.png"))
# print(image_1)

image_dx, image_dy = op.flux_optique(image_1,image_2, 19)
flowpy.show_flow(image_dx,image_dy)
# image_dy = op.derivee_y(image_1)
# plt.subplot(1,2,1)
# plt.imshow(image_dx)
# plt.subplot(1,2,2)
# plt.imshow(image_dy)
# plt.show()