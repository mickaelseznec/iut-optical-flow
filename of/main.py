from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import operators as op

image = np.asarray(Image.open("data/RubberWhale/frame10.png"))
image_dx = op.derivee_x(image)
image_dy = op.derivee_y(image)
plt.subplot(1,2,1)
plt.imshow(image_dx)
plt.subplot(1,2,2)
plt.imshow(image_dy)
plt.show()