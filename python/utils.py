import os
import face_recognition
import cv2
import pytesseract
import time
from math import *
import pytesseract
import numpy as np
import tensorflow as tf
from skimage import segmentation,measure,morphology,color
from skimage import img_as_float
from skimage import img_as_ubyte
from imutils.video import FPS
from utils import *
import model
from icdar import restore_rectangle
import os
import locality_aware_nms as nms_locality


def is_an_image_file(filename):
    '''
    Verfiy if the input file is an image file.
    :param filename: input filename
    :return: A boolean True or False
    '''
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    statinfo = os.stat(filename).st_size
    if (statinfo == 0):
        return False
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def load_paths(file_path):
    '''
    Verify each file in folder if it's an image file, and put all image files in list.
    :param file_path: path to input image folder
    :return: list contains all image path
    '''
    files = os.listdir(file_path)
    return [f for f in files if is_an_image_file(os.path.join(file_path, f))]

def load_paths_and_names(file_path):
    '''
    Load image paths and names of images
    :param file_path: path to input folder
    :return: image_paths: list of images paths, names: list of faces' names(name of image)
    '''
    files = os.listdir(file_path)
    paths = [f for f in files if is_an_image_file(os.path.join(file_path, f))]
    image_paths = [os.path.join(file_path, f) for f in paths]
    names = known_faces_name(paths)
    return image_paths, names

def known_faces_name(paths):
    '''
    Load all faces' names
    :param paths: image paths
    :return: all names in database
    '''
    known_names = []
    for path in paths:
        if path.endswith('.jpg') or path.endswith('.png'):
            known_names.append(path[:-4])
        elif path.endswith('.jpeg'):
            known_names.append(path[:-5])
    return known_names

def create_database(file_path):
    '''
    Create a datebase by loading images and transfer to face encodings
    :param file_path: path to image folder
    :return: known_names: list of names in database, face_images_encoding: list of face encodings
    for each image file
    '''
    paths = load_paths(file_path)
    known_names = known_faces_name(paths)
    image_paths = [os.path.join(file_path, f) for f in paths]
    face_images = [face_recognition.load_image_file(path) for path in image_paths]
    # get the face encodings for each image file
    # Given an image, return the 128-dimension face encoding for each face in the image.
    try:
        face_images_encoding = [face_recognition.face_encodings(face_image)[0] for face_image in face_images]
    except IndexError:
        print("Unable to locate any faces in at least one of the images.")
        quit()
    return known_names, face_images_encoding

def recognize_face(unknown_face_encoding, known_faces_encoding) :
    '''
    Recognize if it's the same face in database
    :param unknown_face_encoding: face encoding to be verified 
    :param known_faces_encoding: list of known face encoding
    :return: the index of face corresponded
    '''
    results = face_recognition.compare_faces(known_faces_encoding, unknown_face_encoding, tolerance=0.45)
    try:
        index = results.index(True)
        return index
    except:
        return len(known_faces_encoding)

def compare_result(expect_result, result):
    '''
    Compare each result for each face to be verified in folder
    :param expect_result: expect result for detected face
    :param result: actual result of detected face
    :return: result and accuracy
    '''
    count = 0
    for i in range(len(result)):
        if result[i] in expect_result[i]:
            count += 1
    return print("Result : %d error out of %d. Accuracy: %.2f" %(len(result) - count, len(result), count/len(result)))

def color_detection(image):
    '''
    Detect certain color in mouth
    :param image: input image
    :return: px, py, pw, ph: (x, y)-coordinate of top-left point, width and height of bounding box
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)

    mask = cv2.inRange(hsv, light_white, dark_white)
    result = cv2.bitwise_and(image, image, mask=mask)
    # threshold image
    ret, threshed_img = cv2.threshold(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY),
                    127, 255, cv2.THRESH_BINARY)
    contours, hier= cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    px, py, pw, ph = 0, 0, 0, 0
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea) 
        # get the bounding rect
        px, py, pw, ph = cv2.boundingRect(c)
    return px, py, pw, ph



def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w = im.shape[:2]

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.14):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('[boxes] {} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = nms_locality.nms_locality(boxes.astype(np.float32), nms_thres)
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes_temp = boxes[boxes[:, 8] > box_thresh]
    if len(boxes_temp) != 0:
        boxes = boxes_temp

    print('[boxes] {} text boxes after nms'.format(len(boxes)))
    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def remove_bound_and_small_area(opencv_img, padding=(0.2, 0.13), min_area=0.1 * 0.1):# padding in (y,x)
    height, width = opencv_img.shape[:2]
    skimage_image = img_as_float(255 - opencv_img)
    labels = measure.label(skimage_image)
    region = measure.regionprops(labels)

    if len(region) == 1:
        out1 = skimage_image
    else:
        num = labels.max()
        del_array = np.array([0] * (num + 1))
        out = skimage_image
        for k in range(num):
            y, x = region[k].centroid
            if padding[1] <= float(x)/width <= 1-padding[1] and padding[0] <= float(y)/height <= 1-padding[0]:
                pass
            else:
                del_array[k+1] = 1 # remove
            if region[k].area < width*height*min_area:
                del_array[k+1] = 1

        del_mask = del_array[labels]
        out = out * del_mask
        out1 = skimage_image - out

    return 255 - img_as_ubyte(out1)


def get_image_in_box(img, box):

    rect = cv2.minAreaRect(box)
    box = cv2.boxPoints(rect)
    (pt1,pt2,pt3,pt4) = box
    angle = rect[2]
    if 0 < abs(angle) and abs(angle) <= 45: # 逆时针
        angle = angle
    elif 45 < abs(angle) and abs(angle) < 90: # 顺时针
        angle = 90 - abs(angle)
    # angle = 0
    # angle-=5

    # cv2.imshow('orig', img)
    # print("orig_angle: {} angle: {}".format(rect[2], angle))

    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]  # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    # cv2.imshow('rotateImg2', imgRotation)
    # cv2.waitKey(0)

    # 旋转后图像的四点坐标
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
    dX = int((endX - startX) * float(0.12))
    dY = int((endY - startY) * float(0.12))

    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(widthNew, endX + (dX * 2))
    endY = min(heightNew, endY + (dY * 2))

    # 裁减得到的旋转矩形框
    imgOut = imgRotation[startY:endY, startX:endX]

    cv2.destroyAllWindows()

    # 放大图像
    imgOut = cv2.resize(imgOut, (imgOut.shape[1] * 2, imgOut.shape[0] * 2))
    #cv2.imshow("resize", imgOut)

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
    #cv2.imshow("erode", imgOut)

    ex_open = cv2.morphologyEx(imgOut, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("open", ex_open)
    # imgOut = ex_open

    # 闭运算
    ex_close = cv2.morphologyEx(imgOut, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("close", ex_close)
    # imgOut = ex_close

    # # 膨胀
    # imgOut = cv2.dilate(imgOut, kernel)
    # cv2.imshow("dilate", imgOut)

    # 平滑
    imgOut = cv2.GaussianBlur(imgOut,(5,5),0)
    #cv2.imshow("GaussianBlur", imgOut)

    # 灰度化
    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("cvtColor", imgOut)

    # 增强对比度
    ehist = cv2.equalizeHist(imgOut)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    chist = clahe.apply(imgOut)
    # cv2.imshow("chist", chist)
    hist = np.hstack((imgOut, ehist, chist))
    #cv2.imshow("hist", hist)

    imgOut = chist
    # imgOut = ehist

    # 膨胀
    imgOut = cv2.dilate(imgOut, kernel)
    #cv2.imshow("dilate", imgOut)

    # # OTSU法二值化
    # ret, th_otsu = cv2.threshold(imgOut, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow("threshold_otsu", th_otsu)
    # imgOut = th_otsu

    # 二值化
    ret, th_bin = cv2.threshold(imgOut, 75, 255, cv2.THRESH_BINARY)
    #cv2.imshow("threshold", th_bin)
    imgOut = th_bin

    # 去除边缘
    imgOut = remove_bound_and_small_area(imgOut)
    #cv2.imshow("remove_bound_area", imgOut)

    cv2.waitKey(0)
    return imgOut

def area_to_char_text(img):
    config = (#10
        "-l eng --oem 0 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    text = pytesseract.image_to_string(img, config=config)
    return text

def area_to_text(img):
    config = (
        "-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    text = pytesseract.image_to_string(img, config=config)
    return text

def split_to_area_and_recognize(opencv_img, padding = (0.08, 0.12)): # padding (y,x)
    height, width = opencv_img.shape[:2]
    skimage_image = img_as_float(255 - opencv_img)
    labels = measure.label(skimage_image)
    regions = measure.regionprops(labels)

    text = ""
    regions.sort(key = lambda r:r.bbox[1])
    dx = int(width * padding[1])
    dy = int(height * padding[0])

    opencv_img_cpy = opencv_img.copy()
    for ind_r, r in enumerate(regions):
        _area = r.bbox
        # text += area_to_char_text(opencv_img[area[0]:area[2], area[1]:area[3]])
        img_to_recog = cv2.copyMakeBorder(255 - img_as_ubyte(r.image), dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        text += area_to_char_text(img_to_recog)
        #'''
        #cv2.imshow("split"+str(ind_r), img_to_recog)
        #'''
        area = [max(0,_area[0]-dy),max(0,_area[1]-dx),min(height,_area[2]+dy),min(width,_area[3]+dx) ]
        cv2.polylines(opencv_img_cpy, [np.array([[area[1],area[0]],[area[3],area[0]],[area[3],area[2]],[area[1],area[2]],[area[1],area[0]]])], True, color=(0, 0, 0), thickness=1)
    #cv2.imshow("split", opencv_img_cpy)
    #cv2.waitKey(0)
    return text


def recognize_to_text(img, box):
    img_in_box = get_image_in_box(img, box)
    text = split_to_area_and_recognize(img_in_box)
    # text = area_to_text(img_in_box);
    return text




