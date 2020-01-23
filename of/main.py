from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("Projet/iut-optical-flow/data/Dimetrodon/frame10.png")

plt.imshow(image, cmap='gray')
plt.show()