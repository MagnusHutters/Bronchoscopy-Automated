import numpy as np





#take a image of ints from 0 to 255 and convert it to a numpy array of floats from 0 to 1 using 32 bit floats
def camImageToNumpy(intImage):
    '''
    Convert an image from a camera to a numpy array of floats from 0 to 1 using 32 bit floats
    Parameters:
    intImage (numpy.ndarray): A NumPy array of shape (H, W, 3) representing an image from a camera.
    
    Returns:
    numpy.ndarray: A numpy array of floats from 0 to 1 using 32 bit floats.
    '''
    return np.array(intImage, dtype=np.float32) / 255.0


def rgb_to_grayscale(rgb_image):
    """
    Convert an RGB image to a grayscale image using the luminosity method,
    assuming input is either in uint8 [0, 255] or float32 [0, 1] format.
    Outputs grayscale image in float32 format normalized between 0 and 1.

    Parameters:
    rgb_image (numpy.ndarray): A NumPy array of shape (H, W, 3) representing an RGB image.

    Returns:
    numpy.ndarray: A grayscale image as a numpy.ndarray in float32 format.
    """
    # Check if the image has an alpha channel and remove it if present
    if rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[..., :3]

    # Apply the luminosity formula
    grayscale = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

    # Normalize to [0, 1] if the input was uint8
    if rgb_image.dtype == np.uint8:
        grayscale /= 255.0

    # Ensure type is float32 (this will do nothing if it's already float32)
    grayscale = grayscale.astype('float32')

    return grayscale