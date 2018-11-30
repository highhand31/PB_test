'''
ref:
https://zhuanlan.zhihu.com/p/32887066
'''

import tensorflow as tf
from tensorflow.python.platform import gfile
import common as cm
from keras.utils import np_utils

model_filename = "model_saver/pb_CLASF_circle__model.pb"

#data preprocess
pic_path = r'.\circle_test'
classifier_num = 2
(x_train, x_train_label, x_test, x_test_label) = cm.data_load(pic_path, train_ratio=0, resize=(64, 64), has_dir=False)
print(x_test_label.shape)
x_test_label = np_utils.to_categorical(x_test_label, classifier_num)
print("test label shape = ", x_test_label.shape)

with tf.Session() as sess:
    with gfile.FastGFile(model_filename, 'rb') as f:

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()

        tf.import_graph_def(graph_def, name='')  #導入計算圖

    sess.run(tf.global_variables_initializer())#將變數初始化

    input_x = sess.graph.get_tensor_by_name("input_x:0")#導入PB黨裡的input_x
    result = sess.graph.get_tensor_by_name("output2:0")

    a = sess.run(result,feed_dict={input_x:x_test[0:4]})
    print(a)
