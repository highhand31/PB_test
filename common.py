# import tensorflow as tf
# from tensorflow.contrib import slim
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt


def data_load(pic_path, train_ratio, resize=None, shuffle=True, normalize=True, has_dir=True):
    labels = {}
    x_train = []
    x_train_label = []
    x_test = []
    x_test_label = []


    if resize is not None:
        width = resize[0]
        height = resize[1]

        print('width = ', width)
        print('height = ', height)

    if train_ratio > 1:
        train_ratio = 1

    if has_dir is True:
        for num, dirs in enumerate(os.scandir(pic_path)):

            if dirs.is_dir():
                print(dirs.name)
                labels[dirs.name] = num
                file_path = os.path.join(pic_path, dirs.name)
                print(file_path)
                files = [file.path for file in os.scandir(file_path) if file.is_file()]
                print("Picture number of dir({}) is {} ".format(file_path, len(files)))
                pic_length = len(files)

                train_num = int(pic_length * train_ratio)
                test_num = pic_length - train_num

                print('train_num = ', train_num)
                print('test_num = ', test_num)

                for pic_num, file in enumerate(files):

                    img = cv2.imread(file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # 圖片進行resize
                    if resize is not None:
                        img = cv2.resize(img, (width, height))
                    if pic_num < train_num:
                        x_train.append(img)
                        x_train_label.append(labels[dirs.name])
                    else:
                        x_test.append(img)
                        x_test_label.append(labels[dirs.name])

    else:#資料夾內沒有資料夾，只有照片

        files = [file.path for file in os.scandir(pic_path) if file.is_file()]
        print("Picture number of dir({}) is {} ".format(pic_path, len(files)))
        pic_length = len(files)
        train_num = int(pic_length * train_ratio)
        test_num = pic_length - train_num
        print('train_num = ', train_num)
        print('test_num = ', test_num)

        for pic_num,file in enumerate(files):
            img = cv2.imread(file)
            # print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 圖片進行resize
            if resize is not None:
                img = cv2.resize(img, (width, height))
            if pic_num < train_num:
                x_train.append(img)
                x_train_label.append(0)
            else:
                x_test.append(img)
                x_test_label.append(0)
            # flatten_num = img.shape[0]*img.shape[1]*img.shape[2]
            # img = img.reshape(flatten_num)

    #-------------------
    x_train = np.array(x_train)
    x_train_label = np.array(x_train_label)
    x_test = np.array(x_test)
    x_test_label = np.array(x_test_label)

    # 將資料進行shuffle
    if shuffle is True:
        indice = np.random.permutation(x_train_label.shape[0])
        x_train = x_train[indice]
        x_train_label = x_train_label[indice]

        indice = np.random.permutation(x_test_label.shape[0])
        x_test = x_test[indice]
        x_test_label = x_test_label[indice]

    if normalize is True:
        x_train = x_train.astype("float32")
        x_train = x_train / 255
        print("normalize")

        x_test = x_test.astype("float32")
        x_test = x_test / 255
        print("normalize")

    print(labels)

    return (x_train, x_train_label, x_test, x_test_label)


def address_get(pic_path, train_ratio, shuffle = True,has_dir=True):
    labels = {}
    x_train_addr = []
    x_test_addr = []
    x_train_label = [];
    x_train_label = np.array(x_train_label)
    x_test_label = [];
    x_test_label = np.array(x_test_label)

    if train_ratio > 1:
        train_ratio = 1

    if has_dir is True:
        for num, dirs in enumerate(os.scandir(pic_path)):

            if dirs.is_dir():
                print(dirs.name)
                labels[dirs.name] = num
                file_path = os.path.join(pic_path, dirs.name)
                print(file_path)
                files = [file.path for file in os.scandir(file_path) if file.is_file()]
                print("Picture number of dir({}) is {} ".format(file_path, len(files)))
                pic_length = len(files)

                train_num = int(pic_length * train_ratio)
                test_num = pic_length - train_num

                print('train_num = ', train_num)
                print('test_num = ', test_num)

                # distribute train data and label address
                x_train_addr.extend(files[:train_num])
                temp = np.full((train_num), labels[dirs.name])
                print(temp.shape)
                x_train_label = np.append(x_train_label, temp)
                print('x_train_addr shape = ', len(x_train_addr))
                print('x_train_label shape = ', x_train_label.shape)

                # distribute test data and label address
                x_test_addr.extend(files[train_num:])
                temp = np.full((test_num), labels[dirs.name])
                print(temp.shape)
                x_test_label = np.append(x_test_label, temp)
                print('x_test_addr shape = ', len(x_test_addr))
                print('x_test_label shape = ', x_test_label.shape)

    else:  # 資料夾內沒有資料夾，只有照片

        files = [file.path for file in os.scandir(pic_path) if file.is_file()]
        print("Picture number of dir({}) is {} ".format(pic_path, len(files)))
        pic_length = len(files)
        train_num = int(pic_length * train_ratio)
        test_num = pic_length - train_num
        print('train_num = ', train_num)
        print('test_num = ', test_num)

        # distribute train data and label address
        x_train_addr.extend(files[:train_num])
        temp = np.zeros(train_num)
        print(temp.shape)
        x_train_label = np.append(x_train_label, temp)
        print('x_train_addr shape = ', len(x_train_addr))
        print('x_train_label shape = ', x_train_label.shape)

        # distribute test data and label address
        x_test_addr.extend(files[train_num:])
        temp = np.zeros(test_num)
        print(temp.shape)
        x_test_label = np.append(x_test_label, temp)
        print('x_test_addr shape = ', len(x_test_addr))
        print('x_test_label shape = ', x_test_label.shape)

    if shuffle is True:
        temp = []
        #x_train_addr = np.array(x_train_addr)
        indice = np.random.permutation(len(x_train_addr))
        # print(indice)
        for index in indice:
            temp.append(x_train_addr[index])

        x_train_addr = temp
        x_train_label = x_train_label[indice]

        #x_test_addr = np.array(x_test_addr)
        temp = []
        indice = np.random.permutation(len(x_test_addr))
        for index in indice:
            temp.append(x_test_addr[index])

        x_test_addr = temp
        x_test_label = x_test_label[indice]

    return (x_train_addr, x_train_label, x_test_addr, x_test_label)