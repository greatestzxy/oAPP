#python detect_area_east.py--test_data_path=111.mp4 --expected=P6

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
detector = dlib.get_frontal_face_detector()
predictor_path = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# create database
known_names, known_faces_encoding = utils.create_database('../Images/database/group3')
known_names.append('unknown')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('expected', '0', '')

gpu_list='0'
box_padding = 0.12

FLAGS = tf.app.flags.FLAGS
checkpoint_path = '../extra/model/east_icdar2015_resnet_v1_50_rbox'

def main(argv=None):
    #fps = 24  # 视频帧率

    #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    #videoWriter = cv2.VideoWriter('E4_1.mp4', fourcc, fps, (1360, 480))  # (1360,480)为视频大小
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
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            print("[start] start processing images")
            #im_fn_list = get_images()
            #vs = cv2.VideoCapture(FLAGS.test_data_path)
            #fps = FPS().start()
            #frame = vs.read()
            #frame = frame.astype(np.float32)
            #print(frame)
            #for im_fn in im_fn_list:
            vs = cv2.VideoCapture(FLAGS.test_data_path)

            fps = FPS().start()


            #fps_ = vs.get(cv2.CV_CAP_PROP_FPS)
            #size = (int(vs.get(cv2.CV_CAP_PROP_FRAME_WIDTH)),
            #        int(vs.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)))


            while True:
                frame = vs.read()
                frame = frame[1]


                if frame is None:
                    break

                frame = imutils.resize(frame, width=1000)
                orig = frame.copy()
                im = frame[:, :, ::-1]
                if(number_correct==0):
                    #print()
                    #print("[image] path: {}".format(frame))
                    #im = cv2.imread(im_fn)[:, :, ::-1]
                    #[:,:,::-1]

                    #frame = frame/256

                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = utils.resize_image(im)
                    #print(im_resized)
                    timer = {'net': 0, 'restore': 0, 'nms': 0}
                    start = time.time()
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                    timer['net'] = time.time() - start

                    boxes, timer = utils.detect(score_map=score, geo_map=geometry, timer=timer)
                    print('[time] net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h

                    duration = time.time() - start_time
                    print('[timing] {}'.format(duration))

                    if boxes is not None:
                        for indBoxes, box in enumerate(boxes):
                            text = utils.recognize_to_text(im[:, :, ::-1], box)
                            # cv2.imwrite('./img_in_box.png', img_in_box)
                            print("[recognize box({})] text: {}".format(indBoxes,text))
                            # to avoid submitting errors
                            box = utils.sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5: # strip small box
                                continue
                            cv2.putText(im[:, :, ::-1], text, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX,2.8,(255, 255, 0), thickness=3)
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=4)
                            _, tmpfilename = os.path.split(FLAGS.test_data_path)
                            cv2.imwrite(os.path.join(labeled_img_dir, tmpfilename + '.labeled.jpg'), im[:, :, ::-1])
                            print('[expected number]',FLAGS.expected)
                            cv2.putText(im[:, :, ::-1], text, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if text == FLAGS.expected:
                                number_correct = 1

                if(number_correct==1):
                        print("number is correct")
                        cv2.putText(im[:, :, ::-1], text, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    print("number is incorrect")





                face_location = face_recognition.face_locations(orig)
                if len(face_location) == 0:
                    pass
                elif len(face_location) > 1:
                    pass
                else:
                    unknown_face_encoding = face_recognition.face_encodings(orig, face_location)[0]
                    index = utils.recognize_face(unknown_face_encoding, known_faces_encoding)
                    name = known_names[index]
                    cv2.putText(im[:, :, ::-1], name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                top, right, bottom, left = face_location[0]
                face_height = bottom - top

                # Draw a box around the face
                cv2.rectangle(im[:, :, ::-1], (left, top), (right, bottom), (0, 0, 255))

                # Display the resulting frame
                # try:
                (x, y, w, h) = mouth_detection.mouth_detection_video(im[:, :, ::-1], detector, predictor)

                if h < 0.2 * face_height:
                    cv2.putText(im[:, :, ::-1], "mouth close,please open your mouth", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                else:
                    cv2.putText(im[:, :, ::-1], "mouth open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    d = int(0.2 * h)
                    roi = orig[y + d:y + h, x:x + w]
                    # cv2.rectangle(frame, (x, y + int(0.2*h)), (x+w, y+h), (0, 255, 0), 2)
                    (px, py, pw, ph) = utils.color_detection(roi)
                    if (pw != 0):
                        cv2.rectangle(im[:, :, ::-1], (x + px, y + d + py), (x + px + pw, y + d + py + ph), (0, 255, 0), 2)
                        if ((pw < w) & (ph < h)):
                            cv2.putText(im[:, :, ::-1],
                                         "please close your mouth for 30 seconds",
                                        (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 0, 255), 2)
                            timer_status = True
                        else:
                            timer_status = False
                            timer = 0
                    else:
                        cv2.putText(im[:, :, ::-1], "no pill detected",
                                    (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        timer_status = False
                        timer = 0
                #videoWriter.write(im[:, :, ::-1])
                fps.update()
                cv2.imshow("im", im[:, :, ::-1])
                key = cv2.waitKey(1) & 0xFF
                #cv2.destroyAllWindows()
                #cv2.imshow("result",orig)
                if key == ord("q"):
                    break

    fps.stop()
    vs.release()
    #videoWriter.release()








if __name__ == '__main__':
    tf.app.run()
