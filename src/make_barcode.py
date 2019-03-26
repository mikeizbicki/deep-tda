import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser


def generate_barcode(data, label):

    """
    Given data generate a rips complex and a
    corresponding barcode

    :params data: numpy array of data
    :returns files: jpeg of barcode
    """

    rips = Rips()

    diagrams = rips.fit_transform(data)
    results = rips.plot(diagrams)
    plt.savefig('images/{0}.png'.format(label))
