import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser, Rips


def generate_barcode(data, label, generate_history=False):

    """
    Given data generate a rips complex and a
    corresponding barcode

    :params data: numpy array of data
    :returns files: jpeg of barcode
    """

    rips = Rips()

    if generate_history == True:
        diagrams = rips.fit_transform(data)
        results = rips.plot(diagrams)
        plt.savefig('images/history-{0}.png'.format(label))

    else:
        img_re_rgb = np.zeros((56,56, 3))
        img_re_rgb[:,:,0] = data

        print("img_re_rgb: ", img_re_rgb)
        plt.imshow(img_re_rgb)
        plt.savefig('images/features-{0}.png'.format(label))

        plt.clf()

        diagrams = rips.fit_transform(data)
        results = rips.plot(diagrams)
        plt.savefig('images/individual-{0}.png'.format(label))

        fig1 = plt.figure()
        plt.clf()
