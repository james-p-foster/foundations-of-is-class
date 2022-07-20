import numpy as np

def angular_difference(angle_array1, angle_array2):
    """
    Calculate the difference between arrays of angle measurements, taking into account the wrapping nature of angles.

    More specifically, if one difference between two angles is larger than pi (half a circle), then there
    actually exists a shorter difference by going  the other way around the circle.
    """
    assert angle_array1.shape == angle_array2.shape

    length = angle_array1.shape[0]
    difference = np.zeros(length)
    for i, (angle1, angle2) in enumerate(zip(angle_array1, angle_array2)):
        if np.abs(angle1) + np.abs(angle2) > np.pi:
            difference[i] = 2*np.pi - (angle2 - angle1)
        else:
            difference[i] = angle2 - angle1
    return difference