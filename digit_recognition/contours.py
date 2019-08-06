from PIL import Image, ImageFilter
import tensorflow as tf
import cv2

#estimate whether rectangular r1 is inside another rectangular r2
def inside(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if (x1 >= x2) and (y1 >= y2) and (x1 + w1 <= x2 + w2) and (y1 + h1 <= y2 + h2):
        return True
    else:
        return False

#get coordinate and size of the rectangular
def wrap_digit(rect):
    #size of the rectangular
    x, y, w, h = rect
    padding = 5
    #horizontal and vertical center of the rectangle
    hcenter = x + w / 2
    vcenter = y + h / 2
    #adjust the rectangular to square
    if (h > w):
        w = h
        x = hcenter - (w / 2)
    else:
        h = w
        y = vcenter - (h / 2)
    return (x - padding, y - padding, w + padding, h + padding)

def imageprocess(path):
    im = Image.open(path).convert('L')
    pixels = list(im.getdata())
    pixel = [ (255-x)*1.0/255.0 for x in pixels]
    return pixel

x_ = tf.placeholder("float", shape=[None, 784])

#initialize parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")

#build CNN
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x_, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

saver = tf.train.Saver()

#set font of the text
font = cv2.FONT_HERSHEY_SIMPLEX

#read the image and convert it to geryscale and apply Gussian filtering
path = "t1.jpg"
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.GaussianBlur(bw, (7, 7), 0)
ret, thbw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV)

#find contours
image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []

#i represents No. of the contour, c represents the contour itself
for i,c in enumerate(cntrs):
    #r / (x,y,w,h) is the smallest box which contains the contour
    r = x, y, w, h = cv2.boundingRect(c)

    # a represents area of the contour
    a = cv2.contourArea(c)

    #img.shape[0]: the width of the picture,img.shape[1] the height of the picture,img.shape[2] how many channels does the picture have
    b = (img.shape[0] - 3) * (img.shape[1] - 3)

    is_inside = False
    #expand the rectangular
    for j,q in enumerate(rectangles):
        if inside(r, q):
            is_inside = True
            break
        # if q is inside r, remove q
        if inside(q, r):
            rectangles.remove(q)
            pass

    if not is_inside:
        if not a == b:
            rectangles.append(r)

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    #restore the parameters of the model
    saver.restore(sess, r'model/train_mnist.model')

    accuracy = tf.argmax(y_conv, 1)

    i = 0
    for r in rectangles:
        i = i + 1
        x, y, w, h = wrap_digit(r)
        #convert x,y,w,h to int
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        #draw the rectangular
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]

        #resize the image
        roi = cv2.resize(roi, (28,28))
        cv2.imwrite("test.png",roi)
        result = imageprocess("test.png")

        #predict the number
        predint = accuracy.eval(feed_dict={x_: [result], keep_prob: 1.0}, session=sess)
        #print the prediction
        cv2.putText(img, "%d" % predint, (x, y - 1), font, 1, (0, 255, 0))

cv2.imshow("contours", img)
cv2.waitKey()