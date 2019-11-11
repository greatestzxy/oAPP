# python face_and_hands.py - -test_data_path=test.mp4 - -expected="E4" - -name="student_A"


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

detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group3')
known_names.append('unknown')

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('expected', '0', '')
tf.app.flags.DEFINE_string('name', '0', '')

FLAGS = tf.app.flags.FLAGS

start_time = datetime.datetime.now()

# max number of hands we want to detect
num_hands_detect = 1

score_thresh = 0.2

timer_start = 0

detection_graph, sess1 = detector_utils.load_inference_graph()

timer = 0

finished = 0


def main(argv=None):

	font_size = 0.6

	font_thickness = 2

	timing = 0

	pill_disappear = 0

	threshold = 5

	gap = 2

	timer2 = 0

	time_up = 0

	t = 1

	nb_fr = 1

	pill_removed = 0

	vs = cv2.VideoCapture(FLAGS.test_data_path)

	length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

	print("frames of the video:" ,length)

	fps = vs.get(cv2.CAP_PROP_FPS)

	print("FPS of the video:   ",fps)

	video_duration = round(length/fps,2)

	print("duration of the video:",video_duration)

	gap = int(fps/2)

	fps = FPS().start()

	all_frames = []

	frame_count = 0



	while True:

		ret, frame = vs.read()

		if ret is False:
			break

		all_frames.append(frame)

		frame_count = frame_count + 1


		if frame_count % gap == 0:

			t = t + 1

			frame = all_frames[frame_count-1]

			frame = imutils.resize(frame, height=1000)

			im = frame[:, :, ::-1]

			orig = frame.copy()
			image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			pill_inside = 0

			mouth_close = 0

			if len(face_recognition.face_locations(orig)) == 0:
				pass
			elif len(face_recognition.face_locations(orig)) > 1:
				pass

			else:
				face_location = face_recognition.face_locations(orig)
				unknown_face_encoding = face_recognition.face_encodings(orig, face_location)[0]
				index = utils.recognize_face(unknown_face_encoding, known_faces_encoding)
				name = known_names[index]
				if name == FLAGS.name:
					nb_fr += 1
				cv2.putText(im[:, :, ::-1], name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)
				top, right, bottom, left = face_location[0]
				#print(top, right, bottom, left)

				face_height = bottom - top
				# Draw a box around the face
				cv2.rectangle(im[:, :, ::-1], (left, top), (right, bottom), (0, 0, 255))
				roi_text = orig[top:bottom, left:right]

				# Display the resulting frame
				# try:
				# if(all(mouth_detection.mouth_detection_video(im[:, :, ::-1], detector, predictor))is not None)::
				try:
					(x, y, w, h) = mouth_detection.mouth_detection_video(orig, detector, predictor)
					cv2.rectangle(im[:, :, ::-1], (x, y), (x + w, y + h), (0, 0, 255))
					d = int(0.35 * h)
					roi = orig[y + d:y + h, x:x + w]
					(px, py, pw, ph) = utils.color_detection_white(roi)  # 检测出白色药片所在坐标

					if (pw!=0):

						cv2.rectangle(im[:, :, ::-1], (x + px, y + py+ d ), (x + px + pw, y + py + ph +d), (0, 255, 0), font_thickness)
						pill_inside = 1

					else:
						pill_inside = 0


					if h < 0.2 * face_height:
						mouth_close = 1
					else:
						mouth_close = 0
				except:
					pass


			boxes1, scores1 = detector_utils.detect_objects(image_np,
			                                                detection_graph, sess1)

			h, w = im.shape[:2]


			hands_detected = detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
					                            scores1, boxes1, w, h,
					                            im[:, :, ::-1])


			if timer2 == 0 & time_up== 0:

				cv2.putText(im[:, :, ::-1], "please put the pill on the tongue and take off your hands",
						            (50, 300),
						            cv2.FONT_HERSHEY_SIMPLEX,
						    font_size, (0, 0, 255),
						    font_thickness)


			if (pill_inside == 1) & (hands_detected == 0) & (time_up == 0) :

					cv2.putText(im[:, :, ::-1], "Please don't remove the pill and close your mouth for 10 seconds",
							            (50, 350), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)
					timer2 = 1


			if timer2 == 1 :
				if (mouth_close == 0) & (pill_inside == 0): # if we can't detect pill when mouth is open, it maybe because of it's the process of closing mouth and the pill is blocked

					pill_disappear = 1

				if (pill_disappear == 1) & (hands_detected == 1):
					cv2.putText(im[:, :, ::-1], "Pill is removed!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, font_size,
							            (0, 0, 255), font_thickness)
					timer2 = 0
					timing = 0
					pill_disappear = 0
					pill_removed = 1


				else:
					if mouth_close==1:
						cv2.putText(im[:, :, ::-1], "Time starts", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)
						timing = timing + 1



			if timing > threshold:
				cv2.putText(im[:, :, ::-1], "Time's up, please open your mouth!", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, font_size,
						            (0, 0, 255), font_thickness)
				time_up = 1

			else:
				time_up = 0


			if time_up == 1:
				if (mouth_close == 0)&(pill_inside==1):
					cv2.putText(im[:, :, ::-1], "Detection is over", (50, 500),cv2.FONT_HERSHEY_SIMPLEX, font_size,(0, 0, 255), font_thickness)
					global finished
					finished = 1


				elif (mouth_close == 0)&(pill_inside==0):
					cv2.putText(im[:, :, ::-1], "Not finished", (50, 500),
							            cv2.FONT_HERSHEY_SIMPLEX, font_size,
							            (0, 0, 255), font_thickness)


			cv2.imshow("im", im[:, :, ::-1])

			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

		else:

			pass



	face_detection_result = nb_fr/t

	if face_detection_result > 0.7:
		right_person = 1
		print("It's the right person")
	else:
		right_person = 0
		print("It's not the right person")

	if finished==1:

		print("Detection finished")

	else:
		print("Detection is not finished")

	fps.stop()
	vs.release()

if __name__ == '__main__':
	tf.app.run()
