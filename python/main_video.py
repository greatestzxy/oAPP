# USAGE
#python test_pill_included.py -v path_to_video

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS

import argparse
import imutils
import time
import cv2
import dlib
import utils
import dlib
import face_recognition
import mouth_detection




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to input video file")
args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group3')
known_names.append('unknown')

print("[INFO] loading facial_landmarks...")


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


timer_status = False
timer = 0
timer_threshold = 30
time_up = False

nb_total = 0
nb_fr = 0
nb_pill = 0

def process_video(input):


	nb_total = 0
	nb_fr = 0
	nb_pill = 0
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

		# resize the frame, this time ignoring aspect ratio

		face_location = face_recognition.face_locations(orig)
		if len(face_location) == 0:
			pass
		elif len(face_location) > 1:
			print("WARNING: two person appear!")
			pass
		else:
			unknown_face_encoding = face_recognition.face_encodings(orig, face_location)[0]
			index = utils.recognize_face(unknown_face_encoding, known_faces_encoding)
			name = known_names[index]
			if name == args["name"]:
				nb_fr += 1
			try:
				top, right, bottom, left = face_location[0]
				face_height = bottom - top
				(x, y, w, h) = mouth_detection.mouth_detection_video(frame, detector, predictor)
				if h > 0.2 * face_height:
					d = int(0.35 * h)
					roi = frame[y + d:y + h, x:x + w]
					(px, py, pw, ph) = utils.color_detection(roi)
					if pw != 0:
						nb_pill += 1
			except:
				pass

			fps.update()
			nb_total += 1

	return nb_total, nb_fr, nb_pill


	# show the output frame
	cv2.imshow("Text Detection", frame)
	key = cv2.waitKey(1) & 0xFF



fps = FPS().start()
nb_total, nb_fr, nb_pill = process_video(args["video"])
fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if nb_pill/nb_total < 0.15:
    print("[INFO] no pill detected : {:.2f}".format(nb_pill/nb_total))
else:
    print("[INFO] pill detected : {:.2f}".format(nb_pill/nb_total))

if nb_fr/nb_total > 0.6:
    print("[INFO] right person : {:.2f}".format(nb_fr/nb_total))
else:
    print("[INFO] wrong person : {:.2f}".format(nb_fr/nb_total))