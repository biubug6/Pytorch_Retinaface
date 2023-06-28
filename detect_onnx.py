from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50, cfg_inference
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import onnxruntime as ort

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--model', default='./FaceDetector.onnx',
                    type=str, help='Path to the ONNX model')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')

args = parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = cfg_inference
    device = torch.device("cpu" if args.cpu else "cuda")
    
    if not os.path.isfile(args.model):
        sys.exit(f'Cannot find the ONNX file {args.model}, exiting')
    model = ort.InferenceSession("FaceDetector.onnx")
    input_shape = model.get_inputs()[0].shape
    print(f'Expected network input shape: {input_shape}')
    batch, colours, im_height, im_width = input_shape

    # Read in the image
    image_path = "./curve/test.jpg"
    img_org = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_org_height, img_org_width, _ = img_org.shape

    # This is the image fed to the network
    img = cv2.resize(np.float32(img_org), (im_height, im_width))

    # Bbox scaling
    scale_bbox = torch.Tensor([img_org_width, img_org_height, img_org_width, img_org_height])
    scale_bbox = scale_bbox.to(device)

    # Landmark scaling
    scale_landms = torch.Tensor([
        img_org_width, img_org_height,
        img_org_width, img_org_height,
        img_org_width, img_org_height,
        img_org_width, img_org_height,
        img_org_width, img_org_height])

    # Normalize the image, reorder colours to planes and add batch size of one
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    tic = time.time()

    loc, conf, landms = model.run(None, {"input0": img.numpy()})
    print('net forward time: {:.4f}'.format(time.time() - tic))

    # These need to be tensors as the rest of the code uses PyTorch
    loc = torch.from_numpy(loc).to(device)
    conf = torch.from_numpy(conf).to(device)
    landms = torch.from_numpy(landms).to(device)

    boxes = loc.squeeze(0)
    boxes = boxes * scale_bbox
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = landms.squeeze(0)
    scale_landms = scale_landms.to(device)
    landms = landms * scale_landms
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # show image
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_org, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_org, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_org, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_org, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_org, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_org, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_org, (b[13], b[14]), 1, (255, 0, 0), 4)
        
        # save image
        name = "test.jpg"
        cv2.imwrite(name, img_org)

