import os, random, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob
import gc
import numpy as np

import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

from utils.data.kidney_dataset import HubDataset


def train(model, train_loader, criterion, optimizer):
    model.train()

    losses = []
    for i, (image, target) in enumerate(train_loader):
        image, target = image.cuda(), target.float().cuda()

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target, 1, False)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.array(losses).mean()


def validation(model, val_loader, criterion):
    model.eval()

    val_probability, val_mask = [], []
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.cuda(), target.float().cuda()

            output = model(image)
            output_ny = output.sigmoid().data.cpu().numpy()
            target_np = target.data.cpu().numpy()

            val_probability.append(output_ny)
            val_mask.append(target_np)

    val_probability = np.concatenate(val_probability)
    val_mask = np.concatenate(val_mask)

    return np_dice_score(val_probability, val_mask)


def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()

    overlap = (p * t).sum()
    dice = 2 * overlap / (uion + 0.001)
    return dice


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


def loss_fn(y_pred, y_true, ratio=0.8, hard=False):
    bce = bce_fn(y_pred, y_true)
    return bce

    # if hard:
    #     dice = dice_fn((y_pred.sigmoid()).float() > 0.5, y_true)
    # else:
    #     dice = dice_fn(y_pred.sigmoid(), y_true)
    # return ratio * bce + (1 - ratio) * dice


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    logging.basicConfig(filename='log.log',
                        format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ',
                        level=logging.INFO)

    DATA_PATH = '/home/fbc/train/kidney/dataset'
    EPOCHES = 50
    BATCH_SIZE = 16
    WINDOW = 1024
    MIN_OVERLAP = 40
    NEW_SIZE = 256

    train_trfm = A.Compose([
        # A.RandomCrop(NEW_SIZE*3, NEW_SIZE*3),
        A.Resize(NEW_SIZE, NEW_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightness(),
            A.ColorJitter(brightness=0.07, contrast=0.07,
                          saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),
        #     A.OneOf([
        #         A.OpticalDistortion(p=0.5),
        #         A.GridDistortion(p=0.5),
        #         A.IAAPiecewiseAffine(p=0.5),
        #     ], p=0.3),
        #     A.ShiftScaleRotate(),
    ])
    val_trfm = A.Compose([
        # A.CenterCrop(NEW_SIZE, NEW_SIZE),
        A.Resize(NEW_SIZE, NEW_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(),
        #     A.OneOf([
        #         A.RandomContrast(),
        #         A.RandomGamma(),
        #         A.RandomBrightness(),
        #         A.ColorJitter(brightness=0.07, contrast=0.07,
        #                    saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        #         ], p=0.3),
        #     A.OneOf([
        #         A.OpticalDistortion(p=0.5),
        #         A.GridDistortion(p=0.5),
        #         A.IAAPiecewiseAffine(p=0.5),
        #     ], p=0.3),
        #     A.ShiftScaleRotate(),
    ])

    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = SoftDiceLoss()

    # 每个file单独做一个验证集
    tiff_ids = np.array([x.split('/')[-1][:-5] for x in glob.glob(os.path.join(DATA_PATH, 'train', '*.tiff'))])


    skf = KFold(n_splits=8)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(tiff_ids, tiff_ids)):
        print(tiff_ids[val_idx])

        # break
        train_ds = HubDataset(DATA_PATH, tiff_ids[train_idx], window=WINDOW, overlap=MIN_OVERLAP,
                              threshold=100, transform=train_trfm)
        valid_ds = HubDataset(DATA_PATH, tiff_ids[val_idx], window=WINDOW, overlap=MIN_OVERLAP,
                              threshold=100, transform=val_trfm, isvalid=False)
        train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
        val_loader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="efficientnet-b1",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pretreined weights for encoder initialization
            in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
        model.cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        # lr_step = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2)
        lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

        header = r'''
                Train | Valid
        Epoch |  Loss |  Dice (Best) | Time
        '''
        print(header)
        #          Epoch         metrics            time
        raw_line = '{:6d}' + '\u2502{:7.4f}' * 3 + '\u2502{:6.2f}'

        best_dice = 0
        for epoch in range(1, EPOCHES + 1):
            start_time = time.time()

            train_loss = train(model, train_loader, loss_fn, optimizer)
            # torch.save(model.state_dict(), 'fold_{}_epoch_{}.pth'.format(fold_idx, epoch))

            val_dice = validation(model, val_loader, loss_fn)
            lr_step.step(val_dice)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), 'fold_{0}_best.pth'.format(fold_idx))

            line = raw_line.format(epoch, train_loss, val_dice, best_dice, (time.time() - start_time) / 60 ** 1)
            print(line)
            logging.info(line)

        del train_loader, val_loader, train_ds, valid_ds
        gc.collect()
