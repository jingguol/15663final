import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
import cv2
import scipy.ndimage
import scipy.io

from Wiener_Filter import wiener_filter


filters = []
for i in range(1, 10) :
    mat = scipy.io.loadmat('../data/filt_scl0' + str(i) + '.mat')
    filter = np.array(mat['filts'][0][4])
    filters.append(filter)

image = plt.imread('../data/trees.bmp')
image = np.divide(image, 255.0, dtype=np.float32)
luminance = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)[:, :, 1]
# plt.imshow(image)
# plt.show()

deconv = np.zeros((9, image.shape[0], image.shape[1], image.shape[2]))
for i in range(9) : 
    deconv[i, :, :, 0] = wiener_filter(image[:, :, 0], filters[i], 1e-2)
    deconv[i, :, :, 1] = wiener_filter(image[:, :, 1], filters[i], 1e-2)
    deconv[i, :, :, 2] = wiener_filter(image[:, :, 2], filters[i], 1e-2)
    # deconv[i, :, :, 0] = skimage.restoration.richardson_lucy(image[:, :, 0], filters[i], num_iter=3)
    # deconv[i, :, :, 1] = skimage.restoration.richardson_lucy(image[:, :, 1], filters[i], num_iter=3)
    # deconv[i, :, :, 2] = skimage.restoration.richardson_lucy(image[:, :, 2], filters[i], num_iter=3)
    deconv[i] = deconv[i] * (np.sum(image) / np.sum(deconv[i]))
    # plt.imshow(deconv[i])
    # plt.show()

reconv = np.zeros(deconv.shape)
for i in range(9) :
    reconv[i, :, :, 0] = scipy.ndimage.convolve(deconv[i, :, :, 0], filters[i], mode='constant', cval=0.0)
    reconv[i, :, :, 1] = scipy.ndimage.convolve(deconv[i, :, :, 1], filters[i], mode='constant', cval=0.0)
    reconv[i, :, :, 2] = scipy.ndimage.convolve(deconv[i, :, :, 2], filters[i], mode='constant', cval=0.0)
    # plt.imshow(reconv[i])
    # plt.show()

energy = np.zeros((9, image.shape[0], image.shape[1]))
for i in range(9) :
    diff = reconv[i] - image
    diff = diff ** 2
    diff = np.sum(diff, axis=2)
    energy[i] = diff
    # plt.imshow(energy[i] / np.max(energy[i]), cmap='gray')
    # plt.show()

windowSize = 25
window = np.ones((windowSize, windowSize), dtype=np.float32)
for i in range(9) :
    energy[i] = scipy.ndimage.convolve(energy[i], window, mode='constant', cval=0.0)
minIdx = np.argmin(energy, axis=0)
plt.imshow(minIdx / np.max(minIdx), cmap='gray')
plt.show()

focused = np.zeros(image.shape)
for i in range(focused.shape[0]) :
    for j in range(focused.shape[1]) :
        focused[i, j, :] = deconv[minIdx[i, j], i, j, :]
# plt.imshow(focused)
# plt.show()
plt.imsave('trees.png', np.clip(focused, 0, 1))