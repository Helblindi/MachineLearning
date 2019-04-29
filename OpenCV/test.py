import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Data/Jessica-ice-cream.JPG', 0)
# https://stackoverflow.com/questions/15072736/extracting-a-region-from-an-image-using-slicing-in-python-opencv/15074748#15074748
# have to convert bgr -> rgb using split/merge
plt.imshow(img, interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
