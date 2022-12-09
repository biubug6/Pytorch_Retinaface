import torch
import torch.optim as optim
from config import config as cfg
# TODO improve Widerface dataset
from dataset.wider_face import WiderFaceDetection, detection_collate
# preproc is the function for performing image augmentation and preprocessing before training
# detection_collate is Custom collate fn for dealing with batches of images that have a different
# number of associated object annotations (bounding boxes)
from dataset.data_augment import preproc
# TODO learn a bit about DataLoader
from torch.utils.data import DataLoader

# TODO improve MultiBoxLoss
from models.multibox_loss import MultiBoxLoss
# TODO improve retinaface model
from models.retinaface import RetinaFace

rgb_mean = (104, 117, 123)
num_classes = 2
img_size = cfg['image_size']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
dataset_path = 'data/train/label.txt'
save_folder = './weights/'
num_workers = 4
momentum = 0.9
weight_decay = 5e-4
initial_lr = 1e-3
gamma = 0.1

net = RetinaFace(cfg=cfg).cuda()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

from utils.prior_box import PriorBox
priorbox = PriorBox(cfg, image_size=(img_size, img_size))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def train():
    dataset = WiderFaceDetection(dataset_path, preproc(img_size, rgb_mean))
    # dataset class inherits torch dataset
    # batchsize number of sample per training step
    # shuffle
    # numworker
    # collate_fn receives a list of tuples if your __getitem__ function from a Dataset subclass returns a tuple, or just
    # or just a normal list if your Dataset subclass returns only one element. Its main objective is to create your batch
    # without spending much time implementing it manually. Try to see it as a glue that you specify the way examples stick
    # together in batch. If you don't use it, Pytorch only put batch_size examples together as you would using torch.stack
    train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
    for epoch in range(max_epoch):
        net.train()
        for iter_index, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['box_weight'] * loss_l + cfg["class_weight"] * loss_c + cfg["landmark_weight"]*loss_landm
            loss.backward()
            optimizer.step()
            print('Epoch:{}, Learning rate:{:.4f}, Loc: {:.4f} Cla: {:.4f} Landm: {:.4f}'
                  .format(epoch, scheduler.get_last_lr()[0], loss_l.item(), loss_c.item(), loss_landm.item()))
            # We need to implement validation here

        scheduler.step()
        torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_{}.pth'.format(str(epoch)))
    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')


if __name__ == '__main__':
    train()
