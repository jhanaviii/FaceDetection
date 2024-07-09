import cv2
import matplotlib.pyplot as plt


def show_image(window_name, image, wait_time=10000):
    """Displays an image in a window and waits for a key press to close it."""
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()


# Face Detection
def detect_faces(image_path, cascade_path):
    """Detects faces in an image and displays the image with rectangles around the faces."""
    image = cv2.imread('/Applications/general/vscode/Projects/PRJ Face Detection/Data Set/people1.jpg')
    image_resized = cv2.resize(image, (800, 600))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Load the face detection model from the XML file
    face_detector = cv2.CascadeClassifier('/Applications/general/vscode/Projects/PRJ Face Detection/Cascades/haarcascade_frontalface_default.xml')
    detections = face_detector.detectMultiScale(image_gray)

    for (x, y, w, h) in detections:
        cv2.rectangle(image_gray, (x, y), (x + w, y + h), (0, 255, 255), 2)

    show_image('Faces Detected', image_gray)


# Car Detection
def detect_cars(image_path, cascade_path):
    """Detects cars in an image and displays the image with rectangles around the cars."""
    image = cv2.imread('/Applications/general/vscode/Projects/PRJ Face Detection/Data Set/car.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the car detection model from the XML file
    car_detector = cv2.CascadeClassifier('/Applications/general/vscode/Projects/PRJ Face Detection/Cascades/cars.xml')
    detections = car_detector.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=8)

    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    show_image('Cars Detected', image)


# Clock Detection
def detect_clocks(image_path, cascade_path):
    """Detects clocks in an image and displays the image with rectangles around the clocks."""
    image = cv2.imread('/Applications/general/vscode/Projects/PRJ Face Detection/Data Set/clock.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the clock detection model from the XML file
    clock_detector = cv2.CascadeClassifier('/Applications/general/vscode/Projects/PRJ Face Detection/Cascades/clocks.xml')
    detections = clock_detector.detectMultiScale(image_gray, scaleFactor=1.03, minNeighbors=1)

    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    show_image('Clocks Detected', image)


# Full Body Detection
def detect_full_bodies(image_path, cascade_path):
    """Detects full bodies in an image and displays the image with rectangles around the bodies."""
    image = cv2.imread('/Applications/general/vscode/Projects/PRJ Face Detection/Data Set/full_bodies.jpg')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the full body detection model from the XML file
    fullbody_detector = cv2.CascadeClassifier('/Applications/general/vscode/Projects/PRJ Face Detection/Cascades/fullbody.xml')
    detections = fullbody_detector.detectMultiScale(image_gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    show_image('Full Bodies Detected', image)


if __name__ == "__main__":

    dataset_dir = '/Applications/general/vscode/Projects/PRJ Face Detection/Data Set'  # Put the images in a folder named 'Data Set' in the same directory as the script
    cascades_dir = '/Applications/general/vscode/Projects/PRJ Face Detection/Cascades'  # Put the XML files in a folder named 'Cascades' in the same directory as the script


    # Face Detection
    detect_faces(f"{dataset_dir}people1.jpg", f"{cascades_dir}haarcascade_frontalface_default.xml")
    detect_faces(f"{dataset_dir}people2.jpg", f"{cascades_dir}haarcascade_frontalface_default.xml")
    detect_faces(f"{dataset_dir}people3.jpg", f"{cascades_dir}haarcascade_frontalface_default.xml")

    # Car Detection
    detect_cars(f"{dataset_dir}car.jpg", f"{cascades_dir}cars.xml")

    # Clock Detection
    detect_clocks(f"{dataset_dir}clock.jpg", f"{cascades_dir}clocks.xml")

    # Full Body Detection
    detect_full_bodies(f"{dataset_dir}full_bodies.jpg", f"{cascades_dir}fullbody.xml")
