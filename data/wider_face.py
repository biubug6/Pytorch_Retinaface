import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from scipy.ndimage import rotate

def cal_rot_cord(image_shape, rotated_img_shape, xy, angle):
    if xy[0] == -1 or xy[1] == -1:
        return np.array((-1, -1))
    org_center = (np.array(image_shape[:2][::-1]) - 1) / 2.
    rot_center = (np.array(rotated_img_shape[:2][::-1]) - 1) / 2.
    org = xy - org_center
    a = np.deg2rad(angle)
    new = np.array([org[0] * np.cos(a) + org[1] * np.sin(a),
                    -org[0] * np.sin(a) + org[1] * np.cos(a)])
    return new + rot_center


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        angle = np.random.randint(0, 90)
        # angle = 0
        img_rotated = rotate(img, angle)
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            pts = [(label[0], label[1]), (label[0] + label[2], label[1] + label[3]), (label[4], label[5]),
                   (label[7], label[8]), (label[10], label[11]), (label[13], label[14]), (label[16], label[17])]
            pts_rt = [cal_rot_cord(img.shape, img_rotated.shape, np.array(pt), angle) for pt in pts]
            # # draw landmarks
            # cv2.circle(img_rotated, (int(pts_rt[2][0]), int(pts_rt[2][1])), 3, (255, 0, 0), 3)
            # cv2.circle(img_rotated, (int(pts_rt[3][0]), int(pts_rt[3][1])), 3, (255, 0, 0), 3)
            # cv2.circle(img_rotated, (int(pts_rt[4][0]), int(pts_rt[4][1])), 3, (255, 0, 0), 3)
            # cv2.circle(img_rotated, (int(pts_rt[5][0]), int(pts_rt[5][1])), 3, (255, 0, 0), 3)
            # cv2.circle(img_rotated, (int(pts_rt[6][0]), int(pts_rt[6][1])), 3, (255, 0, 0), 3)
            # # draw box
            # cv2.circle(img_rotated, (int(pts_rt[0][0]), int(pts_rt[0][1])), 3, (0, 255, 0), 3)
            # cv2.circle(img_rotated, (int(pts_rt[1][0]), int(pts_rt[1][1])), 3, (0, 255, 0), 3)
            # img_rotated = cv2.resize(img_rotated, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # # cv2.rectangle(img_rotated, (int(pts_rt[0][0]), int(pts_rt[0][1])), (int(pts_rt[1][0]), int(pts_rt[1][1])), (255, 0, 0), 1)
            # print(self.imgs_path[index])
            # cv2.imshow(str(self.imgs_path[index]), img_rotated)
            # cv2.waitKey(0)
            # bbox
            annotation[0, 0] = pts_rt[0][0]  # x1
            annotation[0, 1] = pts_rt[0][1]  # y1
            annotation[0, 2] = pts_rt[1][0]  # x2
            annotation[0, 3] = pts_rt[1][1]  # y2

            # landmarks
            annotation[0, 4] = pts_rt[2][0]  # l0_x
            annotation[0, 5] = pts_rt[2][1]  # l0_y
            annotation[0, 6] = pts_rt[3][0]  # l1_x
            annotation[0, 7] = pts_rt[3][1]  # l1_y
            annotation[0, 8] = pts_rt[4][0]  # l2_x
            annotation[0, 9] = pts_rt[4][1]  # l2_y
            annotation[0, 10] = pts_rt[5][0] # l3_x
            annotation[0, 11] = pts_rt[5][1]  # l3_y
            annotation[0, 12] = pts_rt[6][0]  # l4_x
            annotation[0, 13] = pts_rt[6][1]  # l4_y
            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img_rotated, target = self.preproc(img_rotated, target)

        return torch.from_numpy(img_rotated), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
