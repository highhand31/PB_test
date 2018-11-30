import common as cm
import os
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
from tensorflow.python.framework import graph_util


class Classification():
    def __init__(self,classifier_num,input_dim = [None,64,64,3], save_path="model_saver\ckpt"):
        #param setting
        self.input_dim = input_dim
        self.classifier_num = classifier_num
        self.save_path = save_path
        self.dropout_ratio = 0.5

        #data pre-process
        #self.data_init()

        self.__build_model()

    def data_init(self):
        pic_path = r'D:\dataset\xxx'
        (x_train, x_train_label, no1, no2) = cm.data_load(pic_path, train_ratio=1, resize=(width, height), has_dir=True)
        print(x_train.shape)
        print(x_train_label)

    def __build_model(self):
        #input data
        self.input_x = tf.placeholder(tf.float32, self.input_dim, name="input_x")
        self.label_y = tf.placeholder(dtype=tf.float32, shape=[None,self.classifier_num], name="label_y")

        self.prediction = self.__inference(self.input_x)
        self.output = tf.nn.softmax(self.prediction,name="output2")  # 因為在訓練模型裡已經有使用softmax，這邊就不用使用

        #set up Saver
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.saver = tf.train.Saver(max_to_keep=20)
        self.out_dir_prefix = os.path.join(save_path, "model")

        #set up optimizer
        self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_y,
                                                                                  logits=self.prediction),name="loss")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss_function)

        #set up evaluation model
        self.correction_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label_y, 1))
        self.correction_prediction = tf.cast(self.correction_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(self.correction_prediction)

    def __inference(self,input_x):
        self.net = tf.layers.conv2d(
            inputs=input_x,
            filters=32,
            kernel_size=[3, 3],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)

        self.net = tf.layers.max_pooling2d(inputs=self.net, pool_size=[2, 2], strides=2)

        self.net = tf.layers.conv2d(
            inputs=self.net,
            filters=64,
            kernel_size=[3, 3],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)
        self.net = tf.layers.max_pooling2d(inputs=self.net, pool_size=[2, 2], strides=2)

        self.net = tf.layers.conv2d(
            inputs=self.net,
            filters=128,
            kernel_size=[3, 3],
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            padding="same",
            activation=tf.nn.relu)
        self.net = tf.layers.max_pooling2d(inputs=self.net, pool_size=[2, 2], strides=2)


        self.net = tf.layers.flatten(self.net)
        print(self.net)

        #FC1
        self.net = tf.layers.dense(inputs=self.net, units=1024, activation=tf.nn.relu, name='FC1')
        self.net = tf.nn.dropout(self.net, 1 - self.dropout_ratio)

        # FC2
        self.net = tf.layers.dense(inputs=self.net, units=1024, activation=tf.nn.relu, name='FC2')
        self.net = tf.nn.dropout(self.net, 1 - self.dropout_ratio)

        self.net = tf.layers.dense(inputs=self.net, units=self.classifier_num, activation=tf.nn.relu, name='output')

        return self.net

    def train(self, train_data, train_label,test_data, test_label, GPU_ratio=0.2, epochs=50, batch_size=16, fine_tune=False,
              save_ckpt=True):

        # 計算total batch
        total_batches = train_data.shape[0] // batch_size
        if train_data.shape[0] % batch_size:
            total_batches += 1

        # 設定GPU參數
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )
        config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

        with tf.Session(config=config) as sess:
            #confirm ckpt file
            if fine_tune is True:  # 使用已經訓練好的權重繼續訓練
                files = [file.path for file in os.scandir(self.save_path) if file.is_file()]
                if not files:  # 沒有任何之前的權重
                    sess.run(tf.global_variables_initializer())
                    print('no previous model param can be used')
                else:
                    self.saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
                    print('use previous model param')
            else:
                sess.run(tf.global_variables_initializer())
                print('no previous model param can be used')

            #training epochs
            for epoch in range(epochs):
                for index in range(total_batches):

                    num_start = index * batch_size
                    num_end = num_start + batch_size

                    if num_end >= train_data.shape[0] and batch_size > 1:
                        num_end = train_data.shape[0] - 1

                    sess.run(self.optimizer, feed_dict={self.input_x: train_data[num_start:num_end],
                                                        self.label_y:train_label[num_start:num_end]})

                # compute mean loss of train set after a epoch
                train_loss = []
                train_acc = []
                for index in range(train_data.shape[0]):
                    single_loss = sess.run(self.loss_function, feed_dict={self.input_x: train_data[index:index + 1],
                                                                          self.label_y: train_label[index:index + 1]})
                    single_acc = sess.run(self.accuracy, feed_dict={self.input_x: train_data[index:index + 1],
                                                                          self.label_y: train_label[index:index + 1]})

                    train_loss.append(single_loss)
                    train_acc.append(single_acc)

                train_loss = np.array(train_loss)
                train_loss = np.mean(train_loss)
                train_acc = np.array(train_acc)
                train_acc = np.mean(train_acc)

                # compute mean loss of test set after a epoch
                test_loss = []
                test_acc = []
                for index in range(test_data.shape[0]):

                    single_loss = sess.run(self.loss_function, feed_dict={self.input_x: test_data[index:index + 1],
                                                                          self.label_y: test_label[index:index + 1]})
                    single_acc = sess.run(self.accuracy, feed_dict={self.input_x: test_data[index:index + 1],
                                                                          self.label_y: test_label[index:index + 1]})

                    test_loss.append(single_loss)
                    test_acc.append(single_acc)
                test_loss = np.array(test_loss)
                test_loss = np.mean(test_loss)
                test_acc = np.array(test_acc)
                test_acc = np.mean(test_acc)

                msg = "Epoch {}\ntrain set loss = {}, train set accuracy = {}".format(epoch, train_loss,train_acc)
                print(msg)
                #self.UDP_send(msg,self.send_address)

                msg = "test set loss = {}, test set accuracy = {}".format(test_loss, test_acc)
                print(msg)
                #self.UDP_send(msg, self.send_address)

                # 紀錄資料:本次epoch ckpt檔
                if save_ckpt is True:
                    model_save_path = self.saver.save(sess, self.out_dir_prefix, global_step=epoch)

                    print('Save model checkpoint to ', model_save_path)

                graph = tf.get_default_graph().as_graph_def()
                output_graph_def = graph_util.convert_variables_to_constants(sess, graph,['output2'])  # graph也可以直接填入sess.graph_def

                # 'model_saver/'為置放的資料夾，'combined_model.pb'為檔名
                # with tf.gfile.GFile("model_saver/pb_CLASF_circle__model.pb", "wb") as f:
                #     f.write(output_graph_def.SerializeToString())
                #     print("PB file saved successfully")


if __name__ == "__main__":
    save_path = r"model_saver\binary_CLASF_circle"
    out_dir_prefix = os.path.join(save_path, "model")
    height = 64
    width = 64
    epochs = 20
    GPU_ratio = 0.8
    batch_size = 8
    train_ratio = 0.7
    # prepare training data
    pic_path = r'D:\dataset\Halcon_circle'
    classifier_num = 2
    (x_train, x_train_label, x_test, x_test_label) = cm.data_load(pic_path, train_ratio=train_ratio, resize=(width, height), has_dir=True)
    print("x_train shape = ",x_train.shape)
    print("x_test shape = ", x_test.shape)
    x_train_label = np_utils.to_categorical(x_train_label, classifier_num)
    x_test_label = np_utils.to_categorical(x_test_label, classifier_num)
    print("train label shape = ",x_train_label.shape)
    print("test label shape = ", x_test_label.shape)

    CLASF = Classification(classifier_num,input_dim = [None,width,height,3], save_path=save_path)
    CLASF.train(x_train, x_train_label, x_test, x_test_label, GPU_ratio=GPU_ratio, epochs=epochs,
                batch_size=batch_size, fine_tune=True)
