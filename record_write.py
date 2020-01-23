from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.train import Feature
from tensorflow.train import Example
import io
import os
import cv2


def calc_offset(x_start, x_stop, y_start, y_stop, org_size, offset):
    if offset > max(org_size):
        print("ERROR : requirement max(org_size) > offset")
        quit(-1)
    
    h_offset = offset // 2

    x_start_h = x_start - h_offset
    x_stop_h = x_stop + h_offset
    y_start_h = y_start - h_offset
    y_stop_h = y_stop + h_offset

    x_start_offset = -1 * h_offset
    x_stop_offset = h_offset
    y_start_offset = -1 * h_offset
    y_stop_offset = h_offset

    if x_start_h < 0:
        x_start_offset +=h_offset
        x_stop_offset += h_offset 
    if y_start_h < 0:
        y_start_offset += h_offset
        y_stop_offset += h_offset

    if x_stop_h > org_size[0]:
        x_start_offset += -1 * h_offset
        x_stop_offset += -1 * h_offset
    if y_stop_h > org_size[1]:
        y_start_offset += -1 * h_offset
        y_stop_offset += -1 * h_offset

    return  x_start + x_start_offset, x_stop + x_stop_offset, y_start + y_start_offset, y_stop + y_stop_offset

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

def add_num(fname, n):
    base, prefix  = os.path.splitext(fname)
    return base + str(n) + prefix

#def _bytes_feature(value):
#  if isinstance(value, type(tf.constant(0))):
#    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#def _float_feature(value):
#  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#
#def _int64_feature(value):
#  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#def get_feature(example):
#   xmaxs = example.features.feature["image/object/bbox/xmax"].float_list.value
#   ymaxs = example.features.feature["image/object/bbox/ymax"].float_list.value
#   xmins = example.features.feature["image/object/bbox/xmin"].float_list.value
#   ymins = example.features.feature["image/object/bbox/ymin"].float_list.value
#   height = example.features.feature["image/height"].int64_list.value[0]
#   width = example.features.feature["image/width"].int64_list.value[0]
#   image = example.features.feature["image/encoded"].bytes_list.value[0]


def img_split(example, cnt_x, cnt_y, offset=100):

#get example values 
    xmaxs = example.features.feature["image/object/bbox/xmax"].float_list.value
    ymaxs = example.features.feature["image/object/bbox/ymax"].float_list.value
    xmins = example.features.feature["image/object/bbox/xmin"].float_list.value
    ymins = example.features.feature["image/object/bbox/ymin"].float_list.value
    labels = example.features.feature["image/object/class/label"].int64_list.value
    org_height = example.features.feature["image/height"].int64_list.value[0]
    org_width = example.features.feature["image/width"].int64_list.value[0]
    org_fname = example.features.feature["image/filename"].bytes_list.value[0].decode('utf-8')
    image = example.features.feature["image/encoded"].bytes_list.value[0]
    fmt = example.features.feature["image/format"].bytes_list.value[0].decode('utf-8')

    bs = io.BytesIO(image)
    img_pil = Image.open(bs)
    img = pil2cv(img_pil)
    cp_img = img.copy()
    h = img.shape[0]
    w = img.shape[1]

    x_step = (w // cnt_x) 
    y_step = (h // cnt_y) 

    dst_labels = []
    dst_xmaxs = []
    dst_ymaxs = []
    dst_xmins = []
    dst_ymins = []

    for x in range(cnt_x):
        for y in range(cnt_y):

            x_start = (x * x_step) 
            x_stop  = ((x+1) * x_step)
            y_start = (y * y_step) 
            y_stop  = ((y+1) * y_step) 
            org_size = (org_width, org_height)
            x_start, x_stop, y_start, y_stop = calc_offset(x_start, x_stop, y_start, y_stop, org_size, offset)
            sp_img = cp_img.copy()[y_start:y_stop, x_start:x_stop,:]
            for xmax, ymax, xmin, ymin, label in zip(xmaxs, ymaxs, xmins, ymins, labels):
                org_start_x  = int(xmin * org_width)
                org_start_y = int(ymin * org_height)
                org_stop_x = int(xmax * org_width)
                org_stop_y = int(ymax * org_height)
                #cv2.rectangle(img, (org_start_x, org_start_y), (org_stop_x, org_stop_y), (255,0,0),4)

                #in the range
                if (x_start < org_start_x < x_stop) and (y_start < org_start_y < y_stop) and (x_start < org_stop_x < x_stop) and (y_start < org_stop_y < y_stop):
                    cur_w =  org_width 
                    cur_h = org_height
                    start = ( int(cur_w * xmin) - (x*(x_step - offset)), int(cur_h * ymin) - (y * (y_step - offset)))
                    stop =  ( int(cur_w * xmax) - (x*(x_step - offset)), int(cur_h * ymax) - (y * (y_step - offset)))
                    #cv2.rectangle(sp_img, start, stop,(255, 0, 255), 4)

                    dst_xmaxs.append(xmax)
                    dst_ymaxs.append(ymax)
                    dst_xmins.append(xmin)
                    dst_ymins.append(ymin)
                    dst_labels.append(label)
            #fname = 'img_{}_{}.jpg'.format(x,y)
            #cv2.imwrite(fname, sp_img)


###WRITE TF RECORD
            img_str = cv2.imencode('.' + fmt,sp_img)[1].tostring()
            fname = add_num(org_fname, y + (x*cnt_y))
            writer = tf.python_io.TFRecordWriter(os.path.splitext(fname)[0] + '.tfrecord')
            print(fname )
            height = int(y_stop - y_start)
            width =  int(x_stop - x_start)
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
            writer.write(example.SerializeToString())
        #cv2.imwrite("original.jpg", img)

        #f = open("test.jpg", "wb")
        #f.write(img_str)


if __name__ == '__main__':
    record_iterator = tf.python_io.tf_record_iterator('data/test_data.tfrecord')
    
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        img_split(example, 2, 2, offset=100)

  #  writer = tf.python_io.TFRecordWriter("test.tfrecord")
  #key: "image/encoded"
  #key: "image/filename"
  #key: "image/format"
  #key: "image/height"
  #key: "image/width"
  #key: "image/object/bbox/xmax"
  #key: "image/object/bbox/xmin"
  #key: "image/object/bbox/ymax"
  #key: "image/object/bbox/ymin"
  #key: "image/object/class/label"
  #key: "image/object/class/text"
  #key: "image/object/difficult"
  #key: "image/object/truncated"
  #key: "image/source_id"







  #  
  #  height = 100
  #  width = 100
  #  label = 0
  #  example = tf.train.Example(features=tf.train.Features(feature={
  #      'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
  #      'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
  #      'hoge' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
  #      'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))
  #  }))
  #  writer.write(example.SerializeToString())
        #xmaxs = example.features.feature["image/object/bbox/xmax"].float_list.value
        #ymaxs = example.features.feature["image/object/bbox/ymax"].float_list.value
        #xmins = example.features.feature["image/object/bbox/xmin"].float_list.value
        #ymins = example.features.feature["image/object/bbox/ymin"].float_list.value
        #height = example.features.feature["image/height"].int64_list.value[0]
        #width = example.features.feature["image/width"].int64_list.value[0]
        #image = example.features.feature["image/encoded"].bytes_list.value[0]
        #bs = io.BytesIO(image)
        #img_pil = Image.open(bs)
