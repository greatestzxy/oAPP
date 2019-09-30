# USAGE
# python demo.py --east frozen_east_text_detection.pb -v 3.mp4

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import dlib
import utils
import face_recognition
import mouth_detection

detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group3')
known_names.append('unknown')


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
				help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
				help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
				help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
				help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
				help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame, maintaining the aspect ratio
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# resize the frame, this time ignoring aspect ratio
	frame = cv2.resize(frame, (newW, newH))

	# construct a blob from the frame and then perform a forward pass
	# of the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
								 (123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# decode the predictions, then  apply non-maxima suppression to
	# suppress weak, overlapping bounding boxes
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# draw the bounding box on the frame
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

	orig_2 = orig.copy()
	face_location = face_recognition.face_locations(orig_2)
	if len(face_location) == 0:
		pass
	elif len(face_location) > 1:
		pass
	else:
		unknown_face_encoding = face_recognition.face_encodings(orig_2, face_location)[0]
		index = utils.recognize_face(unknown_face_encoding, known_faces_encoding)
		name = known_names[index]
		cv2.putText(orig, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	top, right, bottom, left = face_location[0]
	face_height = bottom - top

	# Draw a box around the face
	cv2.rectangle(orig, (left, top), (right, bottom), (0, 0, 255))

	# Display the resulting frame
	# try:
	(x, y, w, h) = mouth_detection.mouth_detection_video(orig, detector, predictor)

	if h < 0.2 * face_height:
		cv2.putText(orig, "mouth close,please open your mouth", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	else:
		cv2.putText(orig, "mouth open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		d = int(0.35 * h)
		roi = orig[y + d:y + h, x:x + w]  # 在脸的范围内检测
		# cv2.rectangle(frame, (x, y + int(0.2*h)), (x+w, y+h), (0, 255, 0), 2)
		(px, py, pw, ph) = utils.color_detection(roi)  # 检测出白色药片所在坐标
		if (pw != 0):
			cv2.rectangle(orig, (x + px, y + d + py), (x + px + pw, y + d + py + ph), (0, 255, 0), 2)
			if ((pw < w) & (ph < h)):
				cv2.putText(orig, "please close your mouth for 30 seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
							0.7, (0, 0, 255), 2)
				timer_status = True
			else:
				timer_status = False
				timer = 0
		else:
			cv2.putText(orig, "no pill detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			timer_status = False
			timer = 0

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
