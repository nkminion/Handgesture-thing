#Importing libraries and defining variables
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv

ModelPath = '/home/nkminion/Desktop/Python/HandGesture/hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandMarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
Cam = cv.VideoCapture(0)
Timestamp = 0
Increment = 10
WindowName = 'OutputWindow'

#Creating Hand Landmarker

opts = HandLandMarkerOptions(
	base_options = BaseOptions(model_asset_path = ModelPath))

landmarker = HandLandMarker.create_from_options(opts)

# Annotating Landmarks
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        # Draw handedness (left or right hand) on the image.
        cv.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
    return annotated_image

def GetAnnotationFrom(frame):
    Input = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = landmarker.detect(Input)
    annotated_image = draw_landmarks_on_image(Input.numpy_view(), detection_result)
    
    return detection_result, annotated_image

#Feed data and run landmarker
#Open Camera and capture livefeed
while Cam.isOpened():
	ret , frame = Cam.read()
	if not ret:
		break
	DetectionResult , OutputFrame = GetAnnotationFrom(frame)
	cv.imshow(WindowName , OutputFrame)
	c = cv.waitKey(10)
	if (c == 27):
		break
Cam.release()
cv.destroyAllWindows()

