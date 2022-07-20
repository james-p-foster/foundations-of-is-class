import numpy as np

def angular_difference(angle_array1, angle_array2):
    # TODO: do some explaining of the whole enumerate zip and the maths behind this angular difference
    assert angle_array1.shape == angle_array2.shape

    length = angle_array1.shape[0]
    difference = np.zeros(length)
    for i, (angle1, angle2) in enumerate(zip(angle_array1, angle_array2)):
        if np.abs(angle1) + np.abs(angle2) > np.pi:
            difference[i] = 2*np.pi - angle1 - angle2
        else:
            difference[i] = angle2 - angle1
    return difference