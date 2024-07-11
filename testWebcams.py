import cv2

# Function to check and display available webcams
def list_webcams():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        ret, frame = cap.read()
        if ret:
            print(f"Webcam {index} is available")
            # Display the frame from the webcam
            cv2.imshow(f'Webcam {index}', frame)
            print("Press any key to move to the next webcam...")
            cv2.waitKey(0)  # Wait for a key press
            cv2.destroyAllWindows()
        else:
            print(f"Webcam {index} is not available")
        cap.release()
        index += 1

list_webcams()
