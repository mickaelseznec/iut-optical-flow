from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import operators as op
import flowpy
from numba import jit
import time
image1 = np.asarray(Image.open("data/RubberWhale/frame10.png"))
image2 = np.asarray(Image.open("data/RubberWhale/frame11.png"))
start = time.time()
image_dx,image_dy = op.flot_optique(image1,image2,17)
flowpy.show_flow(image_dx,image_dy)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# image_dx = op.derivee_x(image)
# image_dy = op.derivee_y(image)
# plt.subplot(1,2,1)
# plt.imshow(image_dx)
# plt.subplot(1,2,2)
# plt.imshow(image_dy)
# plt.show()