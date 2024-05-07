import numpy as np
import cv2




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


def __prepare_image_basic(image, target_size=(128, 128)): #keep in color mode, and convert to color if neccecary, still do resize and blur
    #convert image to color if neccecary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    #resize the image if neccecary
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
    #blur
    image = cv2.GaussianBlur(image, (7, 7), 0)
    
    
    #convert to float32 in range 0 to 1 if neccesary
    if image.dtype == np.uint8:
        image = image.astype('float32') / 255.0
        
        
    return image
        
    
    
    

def __preprocess_image_no_edge_detection(image, target_size=(128, 128)):
    # Step 1: Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize the image if it's not already the correct size
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Step 3: Apply Gaussian Blur for noise reduction
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # Step 4: Ensure the image is in 0-255 uint8 format if not already
    if image.dtype != np.uint8:
        image = (255 * image).clip(0, 255).astype(np.uint8)
    
    # Step 5: Apply histogram equalization
    image = cv2.equalizeHist(image)

    # Step 6: Convert back to float32 from 0 to 1
    image = image.astype('float32') / 255
    
    return image


def __preprocess_image_CLAHE(image, target_size=(128, 128)):
    # Step 1: Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize the image if it's not already the correct size
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Step 3: Apply Gaussian Blur for noise reduction
    

    # Step 4: Ensure the image is in 0-255 uint8 format if not already
    if image.dtype != np.uint8:
        image = (255 * image).clip(0, 255).astype(np.uint8)
    
    image = cv2.equalizeHist(image)
    # Step 5: Apply histogram equalization
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    #image = clahe.apply(image)
    image = cv2.GaussianBlur(image, (11, 11), 0)

    # Apply CLAHE to the input image

    # Step 6: Convert back to float32 from 0 to 1
    image = image.astype('float32') / 255
    
    return image


def __preprocess_image_with_edge_detection(image, target_size=(128, 128)):
    # Resize the image if necessary
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur for noise reduction
    image = cv2.GaussianBlur(image, (7, 7), 0)
    
    # Check and convert to 0-255 uint8 format if necessary
    if image.dtype != np.uint8:
        image = (255 * image).clip(0, 255).astype(np.uint8)
    
    # Apply histogram equalization
    image = cv2.equalizeHist(image)
    
    # Sobel edge detection in the x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Scale Sobel outputs to 0-1 range and convert to float
    sobel_x = np.clip(sobel_x / 255.0, 0, 1).astype('float32')
    sobel_y = np.clip(sobel_y / 255.0, 0, 1).astype('float32')

    # Stack the channels
    if len(image.shape) == 2:  # Ensure the image is grayscale (not already multi-channel)
        image = image.astype('float32') / 255.0  # Normalize the original image
        image = np.stack((image, sobel_x, sobel_y), axis=-1)

    return image



def __preprocess_image_tresholding(image, target_size=(128, 128)):
    # Step 1: Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize the image if it's not already the correct size
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Step 3: Apply Gaussian Blur for noise reduction
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # Step 4: Ensure the image is in 0-255 uint8 format if not already
    if image.dtype != np.uint8:
        image = (255 * image).clip(0, 255).astype(np.uint8)
    
    # Step 5: Apply histogram equalization
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    image = clahe.apply(image)
    image = cv2.equalizeHist(image)

    #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 2)
    image = cv2.GaussianBlur(image, (7, 7), 0)

    #mean_val = np.mean(image)
    
    # Apply a simple threshold based on the global mean
    _, image1 = cv2.threshold(image, 12, 255, cv2.THRESH_BINARY_INV)
    _, image2 = cv2.threshold(image, 36, 255, cv2.THRESH_BINARY_INV)
    _, image3 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

    # Combine the thresholded images into a single multi-channel image
    image = np.stack((image1, image2, image3), axis=-1)
    
    

    # Step 6: Convert back to float32 from 0 to 1
    image = image.astype('float32') / 255
    
    return image

#preprocess an image wrapper function
def preprocess_image(image, target_size=(128, 128), mode = "none"):
    
    #return __preprocess_image_with_edge_detection(image, target_size)
    #return __preprocess_image_no_edge_detection(image, target_size)
    #return __preprocess_image_tresholding(image, target_size)
    #return __preprocess_image_CLAHE(image, target_size)
    if   mode == "none" or mode == "basic":
        return __prepare_image_basic(image, target_size)
    elif mode == "no_edge":
        return __preprocess_image_no_edge_detection(image, target_size)
    elif mode == "edge":
        return __preprocess_image_with_edge_detection(image, target_size)
    elif mode == "treshold":
        return __preprocess_image_tresholding(image, target_size)
    elif mode == "CLAHE":
        return __preprocess_image_CLAHE(image, target_size)