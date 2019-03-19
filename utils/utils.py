import configparser

import cv2
import numpy as np
import pandas as pd
import torch

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_config(filename):
    answer = dict()
    parser = configparser.ConfigParser()
    parser.read(filename)
    answer["data"] = parser["DIRS"]["DATA_DIR"]
    answer["labels"] = parser["DIRS"]["LABELS_CSV"]

    answer["learning_rate"] = float(parser["LEARNING"]["LEARNING_RATE"])
    answer["batch_size"] = int(parser["LEARNING"]["BATCH_SIZE"])
    answer["batch_epochs"] = int(parser["LEARNING"]["EPOCHS"])
    answer["total_epochs"] = int(parser["LEARNING"]["BATCH_EPOCHS"])
    answer['train_size'] = float(parser["LEARNING"]['TRAIN_SIZE'])

    return answer


def show_image(name, image, waitkey=1):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 480)

    if waitkey > 0:
        cv2.imshow(name, image)
        k = cv2.waitKey(waitkey) & 0xff
    else:
        while True:
            cv2.imshow(name, image)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break


def draw_points(image, keypoints):
    keypoints = keypoints.reshape(-1, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for num, p in enumerate(keypoints):
        cv2.putText(image, "%d" % num, (p[0], p[1]), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
        image = cv2.circle(image, (p[0], p[1]), radius=2, color=(0, 0, 255), thickness=-1)

    show_image("keypoints", image, 0)
    cv2.destroyAllWindows()


def read_points(path):
    points = pd.read_csv(path)
    return points


class Counters:
    def __init__(self):
        self.step = 0
        self.loss = 0

    def update(self, loss):
        self.step += 1
        self.loss = self.loss + 1 / self.step * (loss - self.loss)


def save_model(model, optimizer, train_loss, test_loss, epoch, filename="model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, filename)


def load_model(model, optimizer, filename="model.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    return model, optimizer, train_loss, test_loss, epoch


def train_test_split(data_len, train_size):
    train_size = int(data_len * train_size)
    indx = list(range(data_len))
    np.random.shuffle(indx)
    train_idx, test_idx = indx[:train_size], indx[train_size:]
    return train_idx, test_idx