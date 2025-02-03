#Importing libraries and defining variables
import mediapipe as mp 
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 as cv
import time

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

#Landmarker stuff
class LandmarkerAndResult():
	def __init__(self):
		self.result = vision.HandLandmarkerResult
		self.landmarker = vision.HandLandmarker
		self.createLandmarker()

	def createLandmarker(self):
		#Callback function
		def PrintRes(result: vision.HandLandmarkerResult , output_image: mp.Image , timestamp_ms: int):
			self.result = result

		options = vision.HandLandmarkerOptions(
			base_options = mp.tasks.BaseOptions(model_asset_path=ModelPath),
			running_mode = VisionRunningMode.LIVE_STREAM,
			num_hands = 1,
			min_hand_detection_confidence = 0.3,
			min_hand_presence_confidence = 0.3,
			min_tracking_confidence = 0.3,
			result_callback = PrintRes
		)

		self.landmarker = self.landmarker.create_from_options(options)

	def detect_async(self , frame):
		input = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
		self.landmarker.detect_async(image = input, timestamp_ms = int(time.time() * 1000))

	def close(self):
		self.landmarker.close()
		
def DrawLandmarks(image, DetectionResult: mp.tasks.vision.HandLandmarkerResult):
  try:
    if DetectionResult.hand_landmarks == []: # Empty
      return image
    else:
      HandLandmarksList = DetectionResult.hand_landmarks
      AnnotatedImage = np.copy(image)

      # Loop through the detected hands to visualize.
      for i in range(len(HandLandmarksList)):
        HandLandmarks = HandLandmarksList[i]
          
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in HandLandmarks])
        mp.solutions.drawing_utils.draw_landmarks(
              AnnotatedImage,
              hand_landmarks_proto,
              mp.solutions.hands.HAND_CONNECTIONS,
              mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
              mp.solutions.drawing_styles.get_default_hand_connections_style())
        return AnnotatedImage
  except:
    return image
   
def FingersRaised(image , DetectionResult: mp.tasks.vision.HandLandmarkerResult):
	try:
		# Data
		HandLandMarksList = DetectionResult.hand_landmarks

		Raised = [False , False , False , False , False]
		NumRaised = 0

		for i in range(len(HandLandMarksList)):
			HandLandmarks = HandLandMarksList[i]

			# All fingers excluding thumb
			for j in range (8,21,4):
				TipY = HandLandmarks[j].y
				DipY = HandLandmarks[j-1].y
				PipY = HandLandmarks[j-2].y
				McpY = HandLandmarks[j-3].y
				if TipY < min(DipY,PipY,McpY):
					Raised[int((j/4)-1)] = True
					NumRaised += 1
			#Thumb
			TipX = HandLandmarks[4].x
			DipX = HandLandmarks[3].x 
			McpX = HandLandmarks[1].x 
			PipX = HandLandmarks[2].x
			PalmX =HandLandmarks[0].x 

			#Left
			if (TipX < PalmX):
				if (TipX < min(DipX,PipX,McpX)):
					Raised[0] = True
					NumRaised += 1
			else:
				if (TipX > max(DipX,PipX,McpX)):
					Raised[0] = True
					NumRaised += 1
			
			print("FingerNum: "+str(NumRaised))
			for p in range(5):
				print("Finger "+str(p)+" : "+str(Raised[p]))
	except:
		print("Error")
		
#Open Camera and capture livefeed
def main():
	landmarker = LandmarkerAndResult()
	while Cam.isOpened():
		ret , frame = Cam.read()
		landmarker.detect_async(frame)
		frame = DrawLandmarks(frame,landmarker.result)
		FingersRaised(frame , landmarker.result)
		cv.imshow(WindowName , frame)
		c = cv.waitKey(5)
		if (c == 27):
			break
	landmarker.close()
	Cam.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	main()