config = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "box_weight": 0.5,
    "class_weight": 2.0,
    "landmark_weight": 1.5,
    "batch_size": 8,
    "epoch": 100,
    "image_size": 640,
    "pretrain": False,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256
}

