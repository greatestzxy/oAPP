# python demo.py - -test_data_path=test.mp4 - -expected="E4" - -name = "profA"


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

detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group3')
known_names.append('unknown')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('expected', '0', '')

gpu_list = '0'
box_padding = 0.12

FLAGS = tf.app.flags.FLAGS
checkpoint_path = '../extra/model/east_icdar2015_resnet_v1_50_rbox'

start_time = datetime.datetime.now()
num_frames = 0

# max number of hands we want to detect
num_hands_detect = 2

score_thresh = 0.2

timer_start = 0

detection_graph, sess1 = detector_utils.load_inference_graph()

timer = 0


def main(argv=None):
	# fps = 24  # 视频帧率
	# fourcc = cv2.VideoWriter_fourcc(*'MJPG')

	# videoWriter = cv2.VideoWriter('E4_1.mp4', fourcc, fps, (1360, 480))  # (1360,480)为视频大小
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
	number_correct = 0
	labeled_img_dir = './labeled_image'
	if not os.path.exists(labeled_img_dir):
		os.mkdir(labeled_img_dir)
	number_detected = 0
	number_right = 0

	with tf.get_default_graph().as_default():
		input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

		f_score, f_geometry = model.model(input_images, is_training=False)

		variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
		saver = tf.train.Saver(variable_averages.variables_to_restore())

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
			model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
			saver.restore(sess, model_path)

			print("[start] start processing images")

			vs = cv2.VideoCapture(FLAGS.test_data_path)

			fps = FPS().start()

			n = 0
			pill_disappear = 0
			timer = 0
			while True:
				n = n + 1
				frame = vs.read()
				frame = frame[1]

				if frame is None:
					break


				frame = imutils.resize(frame, width=1000)
				im = frame[:, :, ::-1]
				im2 = frame.copy()
				orig = frame.copy()
				image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

				pill_inside = 0
				hands_detected = 0
				pill_detect = 0
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
					cv2.putText(im[:, :, ::-1], name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					top, right, bottom, left = face_location[0]
					#print(top, right, bottom, left)
					dec = orig[left:right, bottom:top, ::-1]
					face_height = bottom - top
					# Draw a box around the face
					cv2.rectangle(im[:, :, ::-1], (left, top), (right, bottom), (0, 0, 255))
					dec = orig[top:bottom, left:right]

					# Display the resulting frame
					# try:
					# if(all(mouth_detection.mouth_detection_video(im[:, :, ::-1], detector, predictor))is not None)::
					try:
						(x, y, w, h) = mouth_detection.mouth_detection_video(orig, detector, predictor)
						d = int(0.35 * h)
						roi = orig[top:bottom, left:right]  # 在脸的范围内检测
						(px, py, pw, ph) = utils.color_detection_white(roi)  # 检测出白色药片所在坐标

						if (pw != 0):
							cv2.rectangle(im[:, :, ::-1], (left + px, top + py), (left + px + pw, top + py + ph), (0, 255, 0), 2)

							pill_detect = 1

							#if ((x + px + pw) < (x + w)) & ((y + py + ph) < (y + h)):
								#pill_inside = 1

						#if h < 0.2 * face_height:
							#mouth_close = 1
						#else:
							#mouth_close = 0

						else:
							pill_detect = 0


					except:
						pass

				if (number_correct == 0):

					start_time = time.time()
					im_resized, (ratio_h, ratio_w) = utils.resize_image(dec)

					timer = {'net': 0, 'restore': 0, 'nms': 0}
					start = time.time()
					score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
					timer['net'] = time.time() - start

					boxes, timer = utils.detect(score_map=score, geo_map=geometry, timer=timer)
					print('[time] net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(timer['net'] * 1000,
					                                                                   timer['restore'] * 1000,
					                                                                   timer['nms'] * 1000))

					if boxes is not None:
						boxes = boxes[:, :8].reshape((-1, 4, 2))
						boxes[:, :, 0] /= ratio_w
						boxes[:, :, 1] /= ratio_h

					duration = time.time() - start_time
					print('[timing] {}'.format(duration))

					if boxes is not None:
						for indBoxes, box in enumerate(boxes):
							text = utils.recognize_to_text(dec[:, :, ::-1], box)
							# cv2.imwrite('./img_in_box.png', img_in_box)
							print("[recognize box({})] text: {}".format(indBoxes, text))
							# to avoid submitting errors
							box = utils.sort_poly(box.astype(np.int32))
							if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(
									box[3] - box[0]) < 5:  # strip small box
								continue

							add = np.array([[left],[top]])
							ax = box.astype(np.int32).reshape((-1, 1, 2))
							cv2.putText(im[:, :, ::-1], text, (box[0][0]+left,box[0][1]+top), cv2.FONT_HERSHEY_SIMPLEX, 2.8,
							            (255, 255, 0), thickness=3)

							#cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
							#              color=(255, 255, 0), thickness=2)
							_, tmpfilename = os.path.split(FLAGS.test_data_path)
							cv2.imwrite(os.path.join(labeled_img_dir, tmpfilename + '.labeled.jpg'), im[:, :, ::-1])
							print('[expected number]', FLAGS.expected)
							cv2.putText(im[:, :, ::-1], text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
							if text == FLAGS.expected:
								number_correct = 1

				if (number_correct == 1):
					print("number is correct")
					cv2.putText(im[:, :, ::-1], text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				else:
					print("number is incorrect")

				boxes1, scores1 = detector_utils.detect_objects(image_np,
				                                                detection_graph, sess1)
				if boxes1 is not None:
					hands_detected = 1
				else:
					hands_detected = 0

				h, w = im.shape[:2]
				detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
				                                 scores1, boxes1, w, h,
				                                 im[:, :, ::-1])

				if pill_detect == 1:
					cv2.putText(im[:, :, ::-1], "pill detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
					            2)

				else:
					cv2.putText(im[:, :, ::-1], "no pill detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
					            (0, 0, 255), 2)

				if hands_detected == 1:
					cv2.putText(im[:, :, ::-1], "hands detected", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
					            2)

				else:
					cv2.putText(im[:, :, ::-1], "no hands detected", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
					            (0, 0, 255), 2)



				if (pill_inside == 1) :

					cv2.putText(im[:, :, ::-1], "Please put down the hands and show the pill on the tongue",
						        (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				if (pill_inside == 1) & (hands_detected == 0):

						cv2.putText(im[:, :, ::-1], "Please don't remove the pill and close your mouth for 10 seconds",
						            (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						timer = 1



				elif timer ==1:
					if (mouth_close == 0) & (pill_inside == 0):

						pill_disappear = 1

					elif (pill_disappear == 1) & (hands_detected == 1):
						cv2.putText(im[:, :, ::-1], "Pill is removed!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
						            (0, 0, 255), 2)
						timer = 0

				cv2.imshow("im", im[:, :, ::-1])
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
	# cv2.destroyAllWindows()
	# cv2.imshow("result",orig)

	fps.stop()
	vs.release()


# videoWriter.release()


if __name__ == '__main__':
	tf.app.run()