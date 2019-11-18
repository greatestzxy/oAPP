# python number_v1.py - -test_data_path=test.mp4 - -expected="110"


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

timer_start = 0

font_thickness = 2

detection_graph, sess1 = detector_utils.load_inference_graph()

threshold = 10000

num_hands_detect = 1

score_thresh = 0.4



def main(argv=None):

	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
	number_correct = 0
	labeled_img_dir = './labeled_image'

	time1 = time.time()

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

			duration_start = time.time()
			vs = cv2.VideoCapture(FLAGS.test_data_path)

			fps = FPS().start()

			frame_count = 0

			large = 0

			while True:



				not_verified = 1
				processing_time = time.time()-duration_start
				print(processing_time)
				frame_count = frame_count + 1
				frame = vs.read()
				frame = frame[1]

				if frame is None:
					break
				pill_ready = 0
				if large==1:
					frame = imutils.resize(frame, height = 1000)
				else:
					frame = imutils.resize(frame, height=400)
				im = frame[:, :, ::-1]

				orig = frame.copy()
				image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

				boxes1, scores1 = detector_utils.detect_objects(image_np,
				                                                detection_graph, sess1)

				h, w = im.shape[:2]

				hands_detected = detector_utils.draw_box_on_image(num_hands_detect, score_thresh,
				                                                  scores1, boxes1, w, h,
				                                                  im[:, :, ::-1])

				if (processing_time>threshold):
					break


				if hands_detected==0:
					pass

				if len(face_recognition.face_locations(orig)) == 0:
					pass
				elif len(face_recognition.face_locations(orig)) > 1:
					pass

				else:
					face_location = face_recognition.face_locations(orig)

					top, right, bottom, left = face_location[0]

					cv2.rectangle(im[:, :, ::-1], (left, top), (right, bottom), (0, 0, 255))

					roi_text = orig[top:bottom, left:right]

					try:
						(x, y, w, h) = mouth_detection.mouth_detection_video(orig, detector, predictor)
						cv2.rectangle(im[:, :, ::-1], (x, y), (x + w, y + h), (0, 0, 255))
						d = int(0.35 * h)
						roi = orig[y + d:y + h, x:x + w]
						(px, py, pw, ph) = utils.color_detection_white(roi)  # 检测出白色药片所在坐标

						if (pw != 0):

							cv2.rectangle(im[:, :, ::-1], (x + px, y + py + d), (x + px + pw, y + py + ph + d),
							              (0, 255, 0), font_thickness)
							pill_ready = 1

							large = 1

						else:
							pill_ready = 0
					except:
						pass

				if (number_correct == 0)&(pill_ready == 1):

					try:
						print("pill_ready",pill_ready)
						start_time = time.time()
						im_resized, (ratio_h, ratio_w) = utils.resize_image(roi_text)

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
								text = utils.recognize_to_text(roi_text[:, :, ::-1], box)
								if text is not None:
									not_verified = 0

								print("[recognize box({})] text: {}".format(indBoxes, text))

								box = utils.sort_poly(box.astype(np.int32))
								if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(
										box[3] - box[0]) < 5:  # strip small box
									continue

								cv2.putText(im[:, :, ::-1], text, (box[0][0]+left,box[0][1]+top), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
								            (255, 255, 0), thickness=2)

								_, tmpfilename = os.path.split(FLAGS.test_data_path)
								cv2.imwrite(os.path.join(labeled_img_dir, tmpfilename + '.labeled.jpg'), im[:, :, ::-1])
								print('[expected number]', FLAGS.expected)
								cv2.putText(im[:, :, ::-1], text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
								if text == FLAGS.expected:
									number_correct = 1

					except:
						pass

				time1 = time.time() - time1
				time2 = time.time() - duration_start

				if time2>120:
					break


				if (number_correct == 1):

					break




				
				cv2.imshow("im", im[:, :, ::-1])
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break


	fps.stop()
	vs.release()

	if number_correct==1:
		print("number is correct")

	elif (not_verified == 1):
		print("number not verified")

	else:
		print("number is not correct")



	print("whole processing time:",time1)
	print("video processing time",time2)




if __name__ == '__main__':
	tf.app.run()
