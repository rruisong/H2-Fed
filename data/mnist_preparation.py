from genericpath import exists
from sklearn.datasets import fetch_mldata
from tqdm import trange
import numpy as np
import random
import json
import os
import cv2
import shutil
import glob


def train_data1():
    # Setup directory for train/test data
    train_path = 'data/mnist/train/train.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Get MNIST data, normalize, and divide by level
    mnist = fetch_mldata('MNIST original train', data_home='data')
    mu = np.mean(mnist.data.astype(np.float32), 0)
    sigma = np.std(mnist.data.astype(np.float32), 0)
    mnist.data = (mnist.data.astype(np.float32) - mu) / (sigma + 0.001)

    mnist_data = []
    for i in trange(10):
        idx = mnist.target == i
        mnist_data.append(mnist.data[idx])

    print([len(v) for v in mnist_data])

    ###### CREATE USER DATA SPLIT #######
    # Assign 10 samples to each user
    X = [[] for _ in range(110)]
    y = [[] for _ in range(110)]

    idx = np.zeros(10, dtype=np.int64)

    for digit in range(10):
        if digit == 0:
            users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100, 101, 102, 103, 104, 105,
                     106, 107, 108, 109]
        if digit == 1:
            users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                     27, 28, 29]
        if digit == 2:
            users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                     37, 38, 39]
        if digit == 3:
            users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                     47, 48, 49]
        if digit == 4:
            users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                     57, 58, 59]
        if digit == 5:
            users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                     67, 68, 69]
        if digit == 6:
            users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                     77, 78, 79]
        if digit == 7:
            users = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
        if digit == 8:
            users = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        if digit == 9:
            users = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

        for i in range(len(mnist_data[digit])):
            user = np.random.choice(users)
            X[user] += mnist_data[digit][i:i + 1].tolist()
            y[user] += (digit * np.ones(1)).tolist()

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 1000 users
    for i in trange(110, ncols=120):
        uname = 'f_{0:05d}'.format(i)

        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i], 'y': y[i]}
        train_data['num_samples'].append(num_samples)

    print(train_data['num_samples'])
    print(sum(train_data['num_samples']))

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)


def train_data2():
    # Setup directory for train/test data
    train_path = 'data/mnist/train/train.json'

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Get MNIST data, normalize, and divide by level
    mnist = fetch_mldata('MNIST original train', data_home='data')
    mu = np.mean(mnist.data.astype(np.float32), 0)
    sigma = np.std(mnist.data.astype(np.float32), 0)

    mnist_data = []
    for i in trange(10):
        idx = mnist.target == i
        mnist_data.append(mnist.data[idx])

    print([len(v) for v in mnist_data])

    ###### CREATE USER DATA SPLIT #######
    # Assign 10 samples to each user

    for digit in range(10):
        if digit == 0:
            rsu = [0, 1, 9, 10]
        if digit == 1:
            rsu = [0, 1, 2, 10]
        if digit == 2:
            rsu = [0, 1, 2, 3]
        if digit == 3:
            rsu = [0, 2, 3, 4]
        if digit == 4:
            rsu = [0, 3, 4, 5]
        if digit == 5:
            rsu = [0, 4, 5, 6]
        if digit == 6:
            rsu = [0, 5, 6, 7]
        if digit == 7:
            rsu = [6, 7, 8]
        if digit == 8:
            rsu = [7, 8, 9]
        if digit == 9:
            rsu = [8, 9, 10]

        for i in range(len(mnist_data[digit])):
            image = np.reshape(mnist_data[digit][i], (28, 28))
            r = rsu[i % len(rsu)]
            save_path = 'data/mldata/rsu/' + str(r) + '/' + str(digit)
            if not exists(save_path):
                os.makedirs(save_path)
            save_path = save_path + '/' + str(digit) + '_' + str(i) + '.bmp'

            cv2.imwrite(save_path, image)

    for i in range(1, 11):
        for j in range(6):
            if not exists('data/mldata/client/' + str(i * 6 + j)):
                os.makedirs('data/mldata/client/' + str(i * 6 + j))

        digits = os.listdir('data/mldata/rsu/' + str(i))

        digit = digits[0]
        images = os.listdir('data/mldata/rsu/' + str(i) + '/' + str(digit))
        ipc = len(images) // 12
        c0 = images[0:ipc * 6]
        c1 = images[ipc * 6:ipc * 10]
        c3 = images[ipc * 10:ipc * 11]
        c5 = images[ipc * 11:]

        for c in c0:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6) + '/' + c)
        for c in c1:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 1) + '/' + c)
        for c in c3:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 3) + '/' + c)
        for c in c5:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 5) + '/' + c)
        print(len(c0) + len(c1) + len(c3) + len(c5), len(images))

        digit = digits[1]
        images = os.listdir('data/mldata/rsu/' + str(i) + '/' + str(digit))
        ipc = len(images) // 12
        c1 = images[0:ipc]
        c2 = images[ipc:ipc * 7]
        c3 = images[ipc * 7:ipc * 11]
        c5 = images[ipc * 11:]
        for c in c1:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 1) + '/' + c)
        for c in c2:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 2) + '/' + c)
        for c in c3:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 3) + '/' + c)
        for c in c5:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 5) + '/' + c)
        print(len(c1) + len(c2) + len(c3) + len(c5), len(images))

        digit = digits[2]
        images = os.listdir('data/mldata/rsu/' + str(i) + '/' + str(digit))
        ipc = len(images) // 12
        c1 = images[0:ipc]
        c3 = images[ipc:ipc * 2]
        c4 = images[ipc * 2:ipc * 8]
        c5 = images[ipc * 8:]
        for c in c1:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 1) + '/' + c)
        for c in c3:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 3) + '/' + c)
        for c in c4:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 4) + '/' + c)
        for c in c5:
            shutil.copy('data/mldata/rsu/' + str(i) + '/' + str(digit) + '/' + c,
                        'data/mldata/client/' + str(i * 6 + 5) + '/' + c)
        print(len(c1) + len(c3) + len(c4) + len(c5), len(images))

    for j in range(6):
        if not exists('data/mldata/client/' + str(j)):
            os.makedirs('data/mldata/client/' + str(j))

    digits = glob.glob('data/mldata/rsu/0/*/*')

    for i, digit in enumerate(digits):
        shutil.copy(digit, 'data/mldata/client/' + str(i % 6) + '/' + os.path.basename(digit))
        print(i, 'data/mldata/client/' + str(i % 6) + '/' + os.path.basename(digit))

    X = [[] for _ in range(66)]
    y = [[] for _ in range(66)]

    clients = os.listdir('data/mldata/client')

    for user in range(66):
        image_paths = glob.glob('data/mldata/client/' + str(user) + '/*')
        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED).flatten().astype(np.float32)
            image = (image - mu) / (sigma + 0.001)

            digit = int(os.path.basename(path)[0])
            X[user].append(image.tolist())
            y[user] += (digit * np.ones(1)).tolist()

    # Create data structure
    train_data = {'rsu': [], 'user_data': {}, 'num_samples': []}

    # Setup 1000 rsu
    for i in trange(66):
        uname = 'f_{0:05d}'.format((i // 6) * 10 + i % 6)
        print(uname)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])

        train_data['rsu'].append(uname)
        train_data['user_data'][uname] = {'x': X[i], 'y': y[i]}
        train_data['num_samples'].append(num_samples)

    print(train_data['num_samples'])
    print(sum(train_data['num_samples']))

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)


def test_data():
    # Setup directory for train/test data
    test_path = 'data/mnist/test/test.json'

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Get MNIST data, normalize, and divide by level
    mnist = fetch_mldata('MNIST original train', data_home='data')
    mu = np.mean(mnist.data.astype(np.float32), 0)
    sigma = np.std(mnist.data.astype(np.float32), 0)
    mnist = fetch_mldata('MNIST original test', data_home='data')
    mnist.data = (mnist.data.astype(np.float32) - mu) / (sigma + 0.001)

    x = [[] for _ in range(1)]
    y = [[] for _ in range(1)]

    x[0] += mnist.data.tolist()
    y[0] += mnist.target.tolist()

    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    i = 0

    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(x[i], y[i]))
    random.shuffle(combined)
    x[i][:], y[i][:] = zip(*combined)
    num_samples = len(x[i])

    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': x[i], 'y': y[i]}
    test_data['num_samples'].append(num_samples)

    print(test_data['num_samples'])
    print(sum(test_data['num_samples']))

    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == '__main__':
    train_data1()
    test_data()
