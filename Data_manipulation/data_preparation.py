from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os



def crop_image(image_path, cut, row, column):
    img = Image.open(image_path).convert('L')
    img = np.asarray(img.getdata()).reshape((row, column))
    for i in range(row//cut):
        for j in range(column//cut):
            cutimg = img[i*cut:(i+1)*cut]
            cutimg = cutimg[:,list(range(j*cut,(j+1)*cut))]
            plt.imsave('/Users/francescoaldoventurelli/Desktop/birds/{}.jpg'.format(i*(column//cut)+j), cutimg,cmap='gray')

path = "/Users/francescoaldoventurelli/Desktop/QML/MIA_PROVA_QCNN/birds_in_the_sky.jpeg"
crop_image(path, cut=64, row=915, column=1500)

### Images are 260


