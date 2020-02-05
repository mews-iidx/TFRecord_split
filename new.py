import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.train import Feature
from tensorflow.train import Example
import io
import os
import cv2
import sys
import glob
import copy
from offset import split_img

#definitions 
XCNT = 3
YCNT = 3

def usage():
    print('Usage: ' + sys.argv[0] + ' <records_dir_path> <output_dir_path>')

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image
def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def parse_record(input_file):
    record_iterator = tf.python_io.tf_record_iterator(input_file)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        return example.features.feature

def byte2cv(image):
    bs = io.BytesIO(image)
    img_pil = Image.open(bs)
    img = pil2cv(img_pil)
    return img

def is_november(classname):
    ng_classes = ['middle_garbage', 'small_garbage']

    if classname in ng_classes:
        return True
    else:
        return False
def get_instances(parsed_feature):
    classnames = parsed_feature['image/object/class/text'].bytes_list.value
    image      = parsed_feature["image/encoded"].bytes_list.value[0]
    xmaxs      = parsed_feature["image/object/bbox/xmax"].float_list.value
    ymaxs      = parsed_feature["image/object/bbox/ymax"].float_list.value
    xmins      = parsed_feature["image/object/bbox/xmin"].float_list.value
    ymins      = parsed_feature["image/object/bbox/ymin"].float_list.value
    labels     = parsed_feature["image/object/class/label"].int64_list.value
    org_height = parsed_feature["image/height"].int64_list.value[0]
    org_width  = parsed_feature["image/width"].int64_list.value[0]
    org_fname  = parsed_feature["image/filename"].bytes_list.value[0].decode('utf-8')
    fmt        = parsed_feature["image/format"].bytes_list.value[0].decode('utf-8')
    return classnames, image, xmaxs, ymaxs, xmins, ymins, labels, org_height, org_width, org_fname, fmt


def denorm_point(length, points):
    for idx, point in  enumerate(points):
        points[idx] = int(point * length)

def denorm(org_width, org_height, xmaxs, ymaxs, xmins, ymins):
    denorm_point(org_width, xmaxs)
    denorm_point(org_height, ymaxs)
    denorm_point(org_width, xmins)
    denorm_point(org_height, ymins)
    return xmaxs, ymaxs, xmins, ymins

def norm_point(length, points):
    for idx, point in enumerate(points):
        points[idx] = point / length

def norm(org_width, org_height, xmaxs, ymaxs, xmins, ymins):
    norm_point(org_width, xmaxs)
    norm_point(org_height, ymaxs)
    norm_point(org_width, xmins)
    norm_point(org_height, ymins)
    return xmaxs, ymaxs, xmins, ymins
    
def point2child_point(xidx, yidx, xstep, ystep, xmaxs, ymaxs, xmins, ymins):
    for idx, xmax, ymax, xmin, ymin in zip(range(len(xmaxs)), xmaxs, ymaxs, xmins, ymins):
        xmaxs[idx] = int(xmax - (xidx * xstep))
        ymaxs[idx] = int(ymax - (yidx * ystep))
        xmins[idx] = int(xmin - (xidx * xstep))
        ymins[idx] = int(ymin - (yidx * ystep))
    return xmaxs, ymaxs, xmins, ymins

def in_range(point, xmaxs, ymaxs, xmins, ymins, labels):
    dst_xmaxs  = []
    dst_ymaxs  = []
    dst_xmins  = []
    dst_ymins  = []
    dst_labels = []
    for idx, xmax, ymax, xmin, ymin, label in zip(range(len(xmaxs)), xmaxs, ymaxs, xmins, ymins, labels):
        if not point[0] < xmax < point[2]:
            continue
        if not point[1] < ymax < point[3]:
            continue
        if not point[0] < xmin < point[2]:
            continue
        if not point[1] < ymin < point[3]:
            continue
        dst_xmaxs.append(xmax)
        dst_ymaxs.append(ymax)
        dst_xmins.append(xmin)
        dst_ymins.append(ymin)
        dst_labels.append(label)
    return dst_xmaxs, dst_ymaxs, dst_xmins, dst_ymins, dst_labels

def img_encode(fmt, img):
    img_str = cv2.imencode('.' + fmt, img)[1].tostring()
    return img_str

def put_example(height, width, img, fname, fmt, dst_xmaxs, dst_ymaxs, dst_xmins, dst_ymins, dst_labels):
    img_str = img_encode(fmt, img)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "image/width" : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "image/encoded" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str])),
        "image/filename" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[fname.encode('utf-8')])),
        "image/object/bbox/xmax" : tf.train.Feature(float_list = tf.train.FloatList(value=dst_xmaxs)),
        "image/object/bbox/xmin" : tf.train.Feature(float_list = tf.train.FloatList(value=dst_xmins)),
        "image/object/bbox/ymax" : tf.train.Feature(float_list = tf.train.FloatList(value=dst_ymaxs)),
        "image/object/bbox/ymin" : tf.train.Feature(float_list = tf.train.FloatList(value=dst_ymins)),
        "image/object/class/label" : tf.train.Feature(int64_list=tf.train.Int64List(value=dst_labels))
    }))
    return example




def get_imgidx(idx, xcnt, ycnt):
    y = idx // xcnt
    x = idx % xcnt

    return x, y


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 3:
        usage()
        quit()
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    input_files = glob.glob(os.path.join(input_path, '*.tfrecord'))

    for input_file in input_files:
        try:
            parsed_feature = parse_record(input_file) # parse and return
            #TODO: delete line of the below 
            #org_feature = parse_record(input_file)

            #get values
            classnames, bimage, xmaxs, ymaxs, xmins, ymins, labels, org_height, org_width, org_fname, fmt = get_instances(parsed_feature)
        except:
            print("skipping Error file : " + input_file, file=sys.stderr)
            continue
        
        #ignore cases
        ## is no object 
        if len(classnames) == 0 :
            continue
        ## is November data
        decode_name = classnames[0].decode('utf-8')
        if is_november(decode_name):
            print(" skipping class : " + decode_name)
            continue

        #trans norm points to denorm points
        xmaxs, ymaxs, xmins, ymins = denorm(org_width, org_height, xmaxs, ymaxs, xmins, ymins)

        #trans bytes to opencv img
        img = byte2cv(bimage)

        #split image xcnt x ycnt
        xcnt = XCNT
        ycnt = YCNT
        sp_imgs, points = split_img(img, xcnt, ycnt, offset_size=0)

        #split images processing
        for idx, img in enumerate(sp_imgs):
            ystep, xstep, _ = img.shape
            dst_xmaxs, dst_ymaxs, dst_xmins, dst_ymins, dst_labels = in_range(points[idx],xmaxs, ymaxs, xmins, ymins, labels)
            if len(dst_xmaxs) ==0:
                continue

            x, y = get_imgidx(idx, xcnt, ycnt)
            dst_xmaxs, dst_ymaxs, dst_xmins, dst_ymins = point2child_point(x, y, xstep, ystep, dst_xmaxs, dst_ymaxs, dst_xmins, dst_ymins)
            dst_xmaxs, dst_ymaxs, dst_xmins, dst_ymins = norm(xstep, ystep, dst_xmaxs, dst_ymaxs, dst_xmins, dst_ymins)

            #write record
            record_name = os.path.join(output_path, os.path.splitext(os.path.basename(input_file))[0] + str(idx)  + '.tfrecord')
            writer = tf.python_io.TFRecordWriter(record_name)
            example = put_example(ystep, xstep, img, org_fname, fmt, dst_xmaxs, dst_ymaxs, dst_xmins, dst_ymins, dst_labels)
            writer.write(example.SerializeToString())
            print(' saved : ' + record_name)
            print("DEBUG QUIT")
            quit()
