import numpy as np
from scipy import ndimage


def rotate(array, angle):
    """Returns the input array rotated at a given angle."""
    return ndimage.rotate(input=array, angle=angle, order=0, reshape=False)


def flip_h(array):
    """Returns the input array flipped horizontally."""
    return np.fliplr(array)


def flip_v(array):
    """Returns the input array flipped vertically."""
    return np.flipud(array)


def do_nothing(array, **kwargs):
    return array


def flip_and_rotate(array, select, angle):
    """Returns the input array with augmented geometry (flipping
    and rotation only).


    Args:
        array (ndarray): the input array to shift pixels from.
        angle (int): the angle at which the array is rotated.
        select (tuple or list): the selection indices taken randomly
            to specify which augmentation methods to apply.
    """
    # (1) Apply flipping (or not).
    functions = [flip_h, flip_v, do_nothing]
    arr = functions[select[0]](array)

    # (2) Apply rotation (or not).)
    return rotate(arr, angle=angle)
