import cv2
import pathlib
import pandas as pd
import rasterio
from rasterio.windows import Window

import torch.utils.data as D
from torchvision import transforms as T

from .decode import *

identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


class HubDataset(D.Dataset):

    def __init__(self, path, tiff_ids, transform,
                 window=256, overlap=32, threshold=100, isvalid=False):
        self.path = pathlib.Path(path)
        self.tiff_ids = tiff_ids
        self.overlap = overlap
        self.window = window
        self.transform = transform
        self.csv = pd.read_csv((self.path / 'train.csv').as_posix(),
                               index_col=[0])
        self.threshold = threshold
        self.isvalid = isvalid

        self.x, self.y, self.id = [], [], []
        self.build_slices()
        self.len = len(self.x)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def build_slices(self):
        self.masks = []
        self.files = []
        self.slices = []
        for i, filename in enumerate(self.csv.index.values):
            if not filename in self.tiff_ids:
                continue

            filepath = (self.path / 'train' / (filename + '.tiff')).as_posix()
            self.files.append(filepath)

            # print('Transform', filename)
            with rasterio.open(filepath, transform=identity) as dataset:
                self.masks.append(rle_decode(self.csv.loc[filename, 'encoding'], dataset.shape))
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)

                for slc in slices:
                    x1, x2, y1, y2 = slc
                    # print(slc)
                    image = dataset.read([1, 2, 3],
                                         window=Window.from_slices((x1, x2), (y1, y2)))
                    image = np.moveaxis(image, 0, -1)

                    image = cv2.resize(image, (256, 256))
                    masks = cv2.resize(self.masks[-1][x1:x2, y1:y2], (256, 256))

                    if self.isvalid:
                        self.slices.append([i, x1, x2, y1, y2])
                        self.x.append(image)
                        self.y.append(masks)
                        self.id.append(filename)
                    else:
                        if self.masks[-1][x1:x2, y1:y2].sum() >= self.threshold or (image > 32).mean() > 0.99:
                            self.slices.append([i, x1, x2, y1, y2])

                            self.x.append(image)
                            self.y.append(masks)
                            self.id.append(filename)

    # get data operation
    def __getitem__(self, index):
        image, mask = self.x[index], self.y[index]
        augments = self.transform(image=image, mask=mask)
        return self.as_tensor(augments['image']), augments['mask'][None]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
