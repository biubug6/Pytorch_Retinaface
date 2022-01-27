import math
import torch
import torch.optim as optim
import torch.utils.data as data
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
    epoch = 0

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    for iteration in range(0, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers,
                                                  collate_fn=detection_collate))
            if epoch % 10 == 0 and epoch > 0:
                torch.save(net.state_dict(), save_folder + config['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        scheduler.step()
        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = config['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                      epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item()))

    torch.save(net.state_dict(), save_folder + config['name'] + '_Final.pth')


if __name__ == '__main__':
    train()
