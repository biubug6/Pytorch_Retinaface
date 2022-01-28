import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import WiderFaceDetection, detection_collate, preproc, config
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace

rgb_mean = (104, 117, 123)
num_classes = 2
img_dim = config['image_size']
num_gpu = config['ngpu']
batch_size = config['batch_size']
max_epoch = config['epoch']
gpu_train = config['gpu_train']
training_dataset = './data/widerface/train/label.txt'
save_folder = './weights/'
num_workers = 4
momentum = 0.9
weight_decay = 5e-4
initial_lr = 1e-3
gamma = 0.1

net = RetinaFace(cfg=config).cuda()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(config, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def train():
    net.train()
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
    train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
    for epoch in range(max_epoch):
        for iter_index, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = config['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('Epoch:{} Loc: {:.4f} Cla: {:.4f} Landm: {:.4f}'.format(epoch, loss_l.item(), loss_c.item(), loss_landm.item()))
        torch.save(net.state_dict(), save_folder + config['name'] + '_Final.pth')


if __name__ == '__main__':
    train()
