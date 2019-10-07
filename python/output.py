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

def get_image_in_box(img, box):

    rect = cv2.minAreaRect(box)
    box = cv2.boxPoints(rect)
    (pt1,pt2,pt3,pt4) = box
    angle = rect[2]
    if 0 < abs(angle) and abs(angle) <= 45: # 逆时针
        angle = angle
    elif 45 < abs(angle) and abs(angle) < 90: # 顺时针
        angle = 90 - abs(angle)


    # cv2.imshow('orig', img)
    # print("orig_angle: {} angle: {}".format(rect[2], angle))

    height = img.shape[0]  # original height
    width = img.shape[1]  # original width
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # rotate with "angle"
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    # cv2.imshow('rotateImg2', imgRotation)
    # cv2.waitKey(0)

    # coordinate after rotation
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 处理反转的情况
    if pt2[1] > pt4[1]:
        pt2[1], pt4[1] = pt4[1], pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0], pt3[0] = pt3[0], pt1[0]


    # 向外扩展区域
    startY = int(pt2[1])
    endY = int(pt4[1])
    startX = int(pt1[0])
    endX = int(pt3[0])

    # dX = 0
    dX = int((endX - startX) * float(FLAGS.box_padding))
    dY = int((endY - startY) * float(FLAGS.box_padding))

    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(widthNew, endX + (dX * 2))
    endY = min(heightNew, endY + (dY * 2))

    # 裁减得到的旋转矩形框
    imgOut = imgRotation[startY:endY, startX:endX]

    cv2.destroyAllWindows()

    # 放大图像
    imgOut = cv2.resize(imgOut, (imgOut.shape[1] * 2, imgOut.shape[0] * 2))
    cv2.imshow("resize", imgOut)

    print('[INFO] image size: {}'.format(imgOut.shape))

    kernel_size = int(0.04 * min(imgOut.shape[0], imgOut.shape[1]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # ex_tophat = cv2.morphologyEx(imgOut, cv2.MORPH_TOPHAT, kernel)
    # cv2.imshow("tophat", ex_tophat)
    #
    # ex_blackhat = cv2.morphologyEx(imgOut, cv2.MORPH_BLACKHAT, kernel)
    # cv2.imshow("blackhat", ex_blackhat)

    # 腐蚀
    imgOut = cv2.erode(imgOut, kernel)
    cv2.imshow("erode", imgOut)

    ex_open = cv2.morphologyEx(imgOut, cv2.MORPH_OPEN, kernel)
    cv2.imshow("open", ex_open)
    # imgOut = ex_open

    # 闭运算
    ex_close = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("close", ex_close)
    # imgOut = ex_close

    # # 膨胀
    # imgOut = cv2.dilate(imgOut, kernel)
    # cv2.imshow("dilate", imgOut)

    # 平滑
    imgOut = cv2.GaussianBlur(imgOut,(5,5),0)
    cv2.imshow("GaussianBlur", imgOut)

    # 灰度化
    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)
    cv2.imshow("cvtColor", imgOut)

    # 增强对比度
    ehist = cv2.equalizeHist(imgOut)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    chist = clahe.apply(imgOut)
    # cv2.imshow("chist", chist)
    hist = np.hstack((imgOut, ehist, chist))
    cv2.imshow("hist", hist)

    imgOut = chist
    # imgOut = ehist

    # 膨胀
    imgOut = cv2.dilate(imgOut, kernel)
    cv2.imshow("dilate", imgOut)

    # # OTSU法二值化
    # ret, th_otsu = cv2.threshold(imgOut, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow("threshold_otsu", th_otsu)
    # imgOut = th_otsu

    # 二值化
    ret, th_bin = cv2.threshold(imgOut, 75, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold", th_bin)
    imgOut = th_bin

    # 去除边缘
    imgOut = utils.remove_bound_and_small_area(imgOut)
    cv2.imshow("remove_bound_area", imgOut)

    #cv2.waitKey(0)
    return imgOut







def main(argv=None):
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

            #videoWriter = cv2.VideoWriter('22.mp4', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
            while True:
                frame = vs.read()
                frame = frame[1]


                if frame is None:
                    break

                frame = imutils.resize(frame, width=1000)
                orig = frame.copy()
                if(number_correct==0):
                    #print()
                    #print("[image] path: {}".format(frame))
                    #im = cv2.imread(im_fn)[:, :, ::-1]
                    #[:,:,::-1]
                    im = frame[:,:,::-1]
                    #frame = frame/256

                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = utils.resize_image(im)
                    #cv2.imshow("show",im)
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
                            #cv2.imshow("im",im)
                            print('[expected number]',FLAGS.expected)
                            if text == FLAGS.expected:
                                number_correct = 1
                                break

                    if(number_correct==1):
                        print("number is correct")

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
                    cv2.putText(orig, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                top, right, bottom, left = face_location[0]
                face_height = bottom - top

                # Draw a box around the face
                cv2.rectangle(orig, (left, top), (right, bottom), (0, 0, 255))

                # Display the resulting frame
                # try:
                (x, y, w, h) = mouth_detection.mouth_detection_video(orig, detector, predictor)

                if h < 0.2 * face_height:
                    cv2.putText(orig, "mouth close,please open your mouth", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

                else:
                    cv2.putText(orig, "mouth open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    d = int(0.2 * h)
                    roi = orig[y + d:y + h, x:x + w]
                    # cv2.rectangle(frame, (x, y + int(0.2*h)), (x+w, y+h), (0, 255, 0), 2)
                    (px, py, pw, ph) = utils.color_detection(roi)
                    if (pw != 0):
                        cv2.rectangle(orig, (x + px, y + d + py), (x + px + pw, y + d + py + ph), (0, 255, 0), 2)
                        if ((pw < w) & (ph < h)):
                            cv2.putText(orig,
                                         "                                             please close your mouth for 30 seconds",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 0, 255), 2)
                            timer_status = True
                        else:
                            timer_status = False
                            timer = 0
                    else:
                        cv2.putText(orig, "                                                  no pill detected",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        timer_status = False
                        timer = 0
                cv2.imshow("result",orig)
            fps.update()
    fps.stop()
    vs.release()








if __name__ == '__main__':
    tf.app.run()
