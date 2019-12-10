#python part1_face_recognition.py - -test_data_path=test.mp4 - -expected="E4" - -name ="xiyuan"
#To improve the processing accuracy and efficieny, please downsize the frame by changing the parameter in line 153

import cv2
import time
from math import *
import tensorflow as tf
from imutils.video import FPS
import utils
import model
from icdar import restore_rectangle
import os
import locality_aware_nms as nms_locality
from imutils.video import FPS
import numpy as np
import imutils
import dlib
import face_recognition
import mouth_detection
import datetime
import detector_utils
import threading
import sys

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group3')
known_names.append('unknown')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# construct the argument parse and parse the arguments
tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('expected', '0', '')
tf.app.flags.DEFINE_string('name', '0', '')
FLAGS = tf.app.flags.FLAGS

gpu_list = '0'
box_padding = 0.12

checkpoint_path = '../extra/model/east_icdar2015_resnet_v1_50_rbox'

#get the start time
start_time = datetime.datetime.now()

#max number of hands we want to detect
num_hands_detect = 1

score_thresh = 0.4

timer_start = 0

#load hands detection model
detection_graph, sess1 = detector_utils.load_inference_graph()

timer = 0

#flag indicating whether the patient finish the whole process
finished = 0

hands = []

def main(argv=None):


	global im
	process_start = time.time()

	#location of the instructions
	font_location1 = 0
	font_location2 = 25
	font_size = 0.6
	font_thickness = 2
	font_color = (255,255,255)

	timing = 0

	timer2 = 0

	#whether the ten seconds countdown finished
	time_up = 0

	#count how many frames have been processed
	t = 1

	#Count in how many frames we detect the right person
	nb_fr = 1

	vs = cv2.VideoCapture(FLAGS.test_data_path)


	length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

	print("frames of the video:" ,length)

	fps = vs.get(cv2.CAP_PROP_FPS)

	print("FPS of the video:   ",fps)

	video_duration = round(length/fps,2)

	print("duration of the video:",video_duration)

	#process every certain mount(gap) of frame
	gap = 7

	#ten seconds countdown
	threshold = fps / gap *10

	fps = FPS().start()

	frame_count = 0

	#initialize variables
	over=0

	#detect if there's any suspicious action, e.g., remove the pill
	suspicious = 0

	#whether the ten senconds countdown started
	start_counting = 0

	pill_removed = 0

	#patient needs to show the pill number for a certain amount of time(shown_time_threshold)
	shown_time_threshold = threshold / 2
	shown_time = 0

	tolerance = 0

	start_tracking = 0

	while True:

		ret, frame = vs.read()

		if ret is False:
			break

		frame_count = frame_count + 1

		#process every certain amount(gap) of frame
		if frame_count % gap == 0:

			if (over==0) :

				t = t + 1

			frame = imutils.resize(frame, width =450)

			im = frame[:, :, ::-1]

			orig = frame.copy()

			image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			pill_inside = 0

			mouth_close = 0

			# find all the faces and make sure there can not be more than one person
			if len(face_recognition.face_locations(orig)) == 0:
				pass
			elif len(face_recognition.face_locations(orig)) > 1:
				print("WARNING: two person appear!")
				pass

			else:
				face_location = face_recognition.face_locations(orig)
				unknown_face_encoding = face_recognition.face_encodings(orig, face_location)[0]
				index = utils.recognize_face(unknown_face_encoding, known_faces_encoding)
				name = known_names[index]
				if (name == FLAGS.name)&(over==0):
					nb_fr += 1
				cv2.putText(im[:, :, ::-1], name, (font_location1, font_location2), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
				top, right, bottom, left = face_location[0]

				face_height = bottom - top
				# Draw a box around the face
				cv2.rectangle(im[:, :, ::-1], (left, top), (right, bottom), (0, 0, 255))

				try:
					(x, y, w, h) = mouth_detection.mouth_detection_video(orig, detector, predictor)
					cv2.rectangle(im[:, :, ::-1], (x, y), (x + w, y + h), (0, 0, 255))
					d = int(0.35 * h)
					#get the mouth area
					roi = orig[y + d:y + h, x:x + w]
					#detect if there's pill inside the mouth and get the pill location by white color detection in the mouth area
					(px, py, pw, ph) = utils.color_detection_white(roi)

					# pill detected
					if (pw!=0):
						# Draw a box around the pill
						cv2.rectangle(im[:, :, ::-1], (x + px, y + py+ d ), (x + px + pw, y + py + ph +d), (0, 255, 0), font_thickness)
						pill_inside = 1
						start_tracking = 1

					else:
						pill_inside = 0

					#detect whether the mouth is close
					if h < 0.2 * face_height:
						mouth_close = 1
					else:
						mouth_close = 0
						if pill_inside==0 & start_tracking==1:
							suspicious = 1
				except:
					pass

			#detect hands and get the scores of the detected hands
			boxes1, scores1 = detector_utils.detect_objects(image_np,
			                                                detection_graph, sess1)

			h, w = im.shape[:2]

			# draw a box around the hands whose score is greater than the score_thresh
			hands_detected = detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
					                            scores1, boxes1, w, h,
					                            im[:, :, ::-1])

			if (over ==0):
				#step one & two (when the ten seconds count down didn't start or finish):
				if timer2 == 0 & time_up== 0:
					#step one:show the number for certain amount of frames
					if shown_time< shown_time_threshold :
						cv2.putText(im[:, :, ::-1],
						            "Please put the pill in front of your mouth,",
						            (font_location1, font_location2 + 25),
						            cv2.FONT_HERSHEY_SIMPLEX,
						            font_size, font_color,
						            font_thickness)
						cv2.putText(im[:, :, ::-1],
						            "with the number clearly visible to the camera.",
						            (font_location1, font_location2 + 50),
						            cv2.FONT_HERSHEY_SIMPLEX,
						            font_size, font_color,
						            font_thickness)
						#when the pill is hold in front of mouth, start counting
						if (pill_inside==1)&(hands_detected==1):
							shown_time = shown_time + 1

					#Step two:after the number is shown for a certain amount of time
					else:
						cv2.putText(im[:, :, ::-1], "Please put the pill on your tongue,",
						            (font_location1, font_location2 + 75),
						            cv2.FONT_HERSHEY_SIMPLEX,
						            font_size, font_color,
						            font_thickness)
						cv2.putText(im[:, :, ::-1], "then remove your hands.",
						            (font_location1, font_location2 + 100),
						            cv2.FONT_HERSHEY_SIMPLEX,
						            font_size, font_color,
						            font_thickness)


				#Step three:the pill is put inside the mouth(pill is inside the mouth, no hands detected)
				if (pill_inside == 1) & (hands_detected==0) & (time_up == 0) :

						cv2.putText(im[:, :, ::-1], "Please keep the pill on your tongue for 10 seconds",
						            (font_location1, font_location2+125), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
						cv2.putText(im[:, :, ::-1],
						            "with your mouth closed.",
						            (font_location1, font_location2 + 150), cv2.FONT_HERSHEY_SIMPLEX, font_size,
						            font_color, font_thickness)
						#pill is inside the mouth, ten seconds countdown can be strated when the mouth is close
						timer2 = 1


				if timer2 == 1 :
					#if there's hand in the frame during the ten seconds countdown, we assume the patient took the pill out of the mouth
					if  (hands_detected == 1)&(start_counting==1):
						cv2.putText(im[:, :, ::-1], "Please don't remove the pill!", (font_location1, font_location2+175), cv2.FONT_HERSHEY_SIMPLEX, font_size,
						            (0,0,0), font_thickness)
						#reset the ten seconds countdown
						timer2 = 0
						timing = 0
						pill_removed = 1


					else:
						if mouth_close==1:
							cv2.putText(im[:, :, ::-1], "Starting the 10 seconds countdown...", (font_location1, font_location2+200), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
							timing = timing + 1
							start_counting = 1


				#Step four:when the ten seconds countdown is over, patient should open the mouth and show the pill is still on the tongue to make sure he/she didn't took out the pill
				if timing > threshold:
					cv2.putText(im[:, :, ::-1], "Please open your mouth and show", (font_location1, font_location2+225), cv2.FONT_HERSHEY_SIMPLEX, font_size,
					            font_color, font_thickness)
					cv2.putText(im[:, :, ::-1],
					            "the pill is still on your tongue.",
					            (font_location1, font_location2 + 250), cv2.FONT_HERSHEY_SIMPLEX, font_size,
					            font_color, font_thickness)
					time_up = 1

				else:
					time_up = 0


				if time_up == 1:
					# pill is detected inside the mouth
					if (mouth_close == 0)&(pill_inside==1):
						cv2.putText(im[:, :, ::-1], "Thank you. You accomplished all the steps.", (font_location1, font_location2+275),cv2.FONT_HERSHEY_SIMPLEX, font_size,font_color, font_thickness)
						cv2.putText(im[:, :, ::-1],
						            "In two minutes we will verify that the number was correct.",
						            (font_location1, font_location2 + 300), cv2.FONT_HERSHEY_SIMPLEX, font_size,
						            font_color, font_thickness)
						cv2.putText(im[:, :, ::-1],
						            "was correct.",
						            (font_location1, font_location2 + 325), cv2.FONT_HERSHEY_SIMPLEX, font_size,
						            font_color, font_thickness)
						global finished
						#the patient followed all the instruction
						finished = 1
						#detection is finished
						over = 1

					#if we can't detect pill in the mouth, it may because the patient is opening his/her mouth and the pill is blocked
					if(mouth_close==0)&(pill_inside==0):
						tolerance = tolerance + 1

					#if hands show up before the pill is detected, we assume the patient took out the pill
					if (finished==0)&(hands_detected==1):
						#the patient didn't follow all the instructions
						finished = 0
						#detection is finished
						over = 1

					elif (tolerance > 3):
						over = 1


			else:
				cv2.putText(im[:, :, ::-1], "Thank you. You accomplished all the steps.",
				            (font_location1, font_location2 + 275), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color,
				            font_thickness)
				cv2.putText(im[:, :, ::-1],
				            "In two minutes we will verify that the number",
				            (font_location1, font_location2 + 300), cv2.FONT_HERSHEY_SIMPLEX, font_size,
				            font_color, font_thickness)
				cv2.putText(im[:, :, ::-1],
				            "was correct.",
				            (font_location1, font_location2 + 325), cv2.FONT_HERSHEY_SIMPLEX, font_size,
				            font_color, font_thickness)

			cv2.imshow("im", im[:, :, ::-1])

			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

		else:
			pass

	cv2.waitKey(0)

	face_detection_result = nb_fr/t

	#calculate the processing time
	process_time = time.time() - process_start

	if face_detection_result > 0.6:
		right_person = 1
		print("It's the right person")
		sys.stdout.flush()
	else:
		right_person = 0
		print("It's not the right person")
		sys.stdout.flush()

	if finished == 1:
		print("Detection finished")
		sys.stdout.flush()
	else:
		print("Detection is not finished")
		sys.stdout.flush()


	if pill_removed==1:
		print("pill has been removed")
		sys.stdout.flush()

	print("process time:",process_time)
	print("suspicious",suspicious)
	#print("Video length:",video_duration)

	#print(face_detection_result)

	fps.stop()
	vs.release()

if __name__ == '__main__':
	tf.app.run()
