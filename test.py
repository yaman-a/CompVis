from skimage import io
import numpy as np
import math


img_path = 'images/mandrill.jpg'

img = io.imread(img_path, as_gray=False)

imgarr = np.array(img)

imgarr = np.divide(imgarr, 255)

# print(imgarr)

