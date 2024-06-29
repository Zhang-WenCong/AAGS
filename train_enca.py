from scene.ha_networks import E_attr
import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import glob
import pandas as pd

class DynamicResize:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, img):
        width, height = img.size
        new_width = int(width / self.scale_factor)
        new_height = int(height / self.scale_factor)
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        return img

class CustomTripletDataset(Dataset):
    def __init__(self, root_dir, pos_data_augmentation, neg_data_augmentation):
        self.image_dir = root_dir + "/images"
        self.root_dir = root_dir
        self.updown_scale = DynamicResize(16)
        self.pos_data_augmentation = pos_data_augmentation
        self.neg_data_augmentation = neg_data_augmentation
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        tsv = glob.glob(os.path.join(root_dir, '*.tsv'))[0]
        files = pd.read_csv(tsv, sep='\t')
        # files = files[~files['id'].isnull()] # remove data without id
        # files.reset_index(inplace=True, drop=True)
        # files = files[files['split']=='train']
        self.image_filenames = [os.path.join(self.image_dir, f) for f in files['filename']]
        print(f"using wild dataset, {len(self.image_filenames)} train images")

        self.images = []
        self._create_triplet()
        

    def __len__(self):
        return 736

    def __getitem__(self, idx):
        anchor_image = self.images[idx]

        positive_image = self.pos_data_augmentation(anchor_image)
        negative_image = self.neg_data_augmentation(anchor_image)
        anchor_image = self.normalize(anchor_image)

        return anchor_image, positive_image, negative_image


    def _create_triplet(self):
        print("begin load img")
        for path in self.image_filenames:
            self.images.append(self.toTensor(self.updown_scale(Image.open(path).convert('RGB'))))
        print("end load img")

if __name__ == '__main__':
    # 数据增强
    pos_data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        # transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    neg_data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, [0.1, 0.45]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    triplet_dataset = CustomTripletDataset('/data/dataset/nerf_wild/brandenburg_gate/dense', pos_data_augmentation, neg_data_augmentation)
    dataloader = DataLoader(triplet_dataset, batch_size=None, shuffle=True)
    enc_a = E_attr(3, 48).cuda()
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(enc_a.parameters(), lr=5e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20], 0.1)
    loss = 0
    for epoch in range(20):
        for i, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.cuda().unsqueeze(0), positive.cuda().unsqueeze(0), negative.cuda().unsqueeze(0)
            optimizer.zero_grad()

            anchor_out = enc_a(anchor)
            positive_out = enc_a(positive)
            negative_out = enc_a(negative)
            sq_sum = (torch.square(anchor_out).mean()+torch.square(positive_out).mean()+torch.square(negative_out).mean()) / 3
            loss += criterion(anchor_out, positive_out, negative_out) + 1e-2 * sq_sum
            if (i+1) % 32 == 0:   
                loss /= 32
                loss.backward()
                optimizer.step()
                scheduler.step()
                print(f"Epoch [{epoch+1}/{20}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                loss = 0

    torch.save(enc_a.state_dict(), './weights/enc_a.pth')