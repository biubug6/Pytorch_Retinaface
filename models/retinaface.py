import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH
from utils.box_utils import decode_tensors, decode_landm_tensors
from layers.functions.prior_box import PriorBox


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train', image_size = None):
        """
        Parameters
        ----------
        cfg : dict
            A dictionary containing the configuration parameters
        phase : str
            Phase of the model, possible values are: train, test and test-decode
            In test-decode mode, both bboxes and landmarks from the network are decoded
            so that the coordinates are scaled between 0..1. In this mode the image_size
            needs to be given.
        image_size: pair
            Image size is needed for decoding coordinates, required if test-decode mode.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

        if phase == 'test-decode':
            # Calculate the parameters required for decoding bboxes and landmarks, and store
            # the values for later use.
            if image_size is None:
                print('Image size needs to be given in onnx-phase')
            priorbox = PriorBox(cfg, image_size)
            cx, cy, s_kx, s_ky = priorbox.create_anchors(separated_variables=True)
            self.register_buffer('cx', torch.tensor(cx, dtype=torch.float32).unsqueeze(0))
            self.register_buffer('cy', torch.tensor(cy, dtype=torch.float32).unsqueeze(0))
            self.register_buffer('s_kx', torch.tensor(s_kx, dtype=torch.float32).unsqueeze(0))
            self.register_buffer('s_ky', torch.tensor(s_ky, dtype=torch.float32).unsqueeze(0))
            self.register_buffer('variances', torch.tensor(cfg['variance'], dtype=torch.float32))
            del priorbox, cx, cy, s_kx, s_ky

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        #bbox_regressions = torch.arange(1*16800*4, dtype=torch.float32).reshape((1, 16800, 4))

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        elif self.phase == 'test':
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        elif self.phase == 'test-decode':
            output = (
                decode_tensors(bbox_regressions, self.cx, self.cy, self.s_kx, self.s_ky, self.variances),
                F.softmax(classifications, dim=-1),
                decode_landm_tensors(ldm_regressions, self.cx, self.cy, self.s_kx, self.s_ky, self.variances)
            )
        else:
            print("Possible values for phase are: train, text or test-decode")
            return -1
        return output
