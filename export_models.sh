#!/bin/bash

python3 export.py -m weights/mobilenet0.25_Final.pth --network mobile0.25 --mode tscript mnet.0.25.pt
python3 export.py -m weights/Resnet50_Final.pth --network resnet50 --mode tscript resnet50.pt

python3 export.py -m weights/mobilenet0.25_Final.pth --network mobile0.25 mnet.0.25.onnx
python3 export.py -m weights/Resnet50_Final.pth --network resnet50 resnet50.onnx
