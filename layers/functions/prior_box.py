import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        print(cfg)
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def create_anchors(self, separated_variables=False):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        if separated_variables:
            anchors = np.array(anchors).reshape(-1, 4)
            cx = anchors[:, 0].copy().reshape(-1, 1)
            cy = anchors[:, 1].copy().reshape(-1, 1)
            s_kx = anchors[:, 2].copy().reshape(-1, 1)
            s_ky = anchors[:, 3].copy().reshape(-1, 1)
            del anchors
            return cx, cy, s_kx, s_ky
        else:
            return anchors

    def forward(self):
        # Torch
        output = torch.Tensor(self.create_anchors()).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
