from PIL import Image
import os
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input = sample['input']
        input = input.transpose((2, 0, 1)) / 255.0

        if 'target' in sample:
            target = sample['target'].astype(np.float32)
            return {
                'input': torch.from_numpy(input).float(),
                'target': torch.from_numpy(target).unsqueeze(0)
            }
        else:
            return {'input': torch.from_numpy(input).float()}

class WaterLevelDataset(torch.utils.data.Dataset):
    """Water Level dataset."""

    def __init__(self, imgs, ann_imgs, transform=None):
        self.imgs = imgs
        self.ann_imgs = ann_imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = {'input': self.imgs[idx], 'target': self.ann_imgs[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

class WT3Net(nn.Module):
    def __init__(self, num_channels, filter_size=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_channels, filter_size, padding='same')
        self.conv2 = nn.Conv2d(num_channels, num_channels, filter_size, padding='same')
        self.conv3 = nn.Conv2d(num_channels, 1, 1)
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))  # Using sigmoid for binary output
        return x

def get_annotation_mask(annotated):
    annotation_mask = (annotated[:, :, 0] / (annotated[:, :, 1] + 1e-10) > 1.8) * (annotated[:, :, 0] / (1e-10 + annotated[:, :, 2]) > 1.8)
    annotation_mask[0:50] = 0 
    annotation_mask[150:] = 0
    return annotation_mask

def pull_training_data(DATADIR):
    filenames = [os.path.join(DATADIR, _) for _ in os.listdir(DATADIR)
                 if os.path.isfile(os.path.join(DATADIR, _)) and _.endswith('.bmp')]
    annotated_filenames = [_.replace('.bmp', ' copy.jpeg') for _ in filenames]
    imgs, ann_imgs = [], []
    for f_, a_f_ in zip(filenames, annotated_filenames):
        original = np.asarray(Image.open(f_))[:, :, :3]
        annotated = np.asarray(Image.open(a_f_))
        annotation_mask = get_annotation_mask(annotated)
        imgs.append(original)
        ann_imgs.append(annotation_mask)
    return imgs, ann_imgs

def train_model(args, dataset):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    model = WT3Net(num_channels=10, filter_size=5) #filter variable here
    criterion = nn.BCELoss()  # Using BCELoss for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    max_epochs = args['max_epochs']

    for epoch_idx in range(max_epochs):
        loss_list = []
        for dta in trainloader:
            inputs = dta['input']
            targets = dta['target']
            outp = model(inputs)
            loss = criterion(outp, targets)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch_idx + 1) % 5 == 0:
            print(f'Epoch {epoch_idx + 1}/{max_epochs} Loss: {np.mean(loss_list)}')

    model_path = os.path.join(args['namespace_dir'], 'model.pt')
    torch.save(model.state_dict(), model_path)
    return model

def generate_output(dataset, model, outdir, T=1e-2):
    for idx in range(len(dataset)):
        dta = dataset[idx]
        inputs = dta['input'].unsqueeze(0)  # Add batch dimension
        outp = model(inputs).squeeze(0).cpu().detach().numpy()
        img = inputs.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))

        # Rescale the output and overlay
        img[:, :, 0] += outp[0]
        img[:, :, 0] = img[:, :, 0] / img[:, :, 0].max()
        plt.imsave(os.path.join(outdir, f'img_{idx}.jpg'), img)
        plt.imsave(os.path.join(outdir, f'img_gray_{idx}.jpg'), outp[0], cmap='gray')

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir", type=Path, default=os.path.join(dir_path, "data/annotated_bmp"), help="Path to the training data directory")
    parser.add_argument('--max_epochs', type=int, default=75)
    parser.add_argument('-n', '--namespace', type=str, default='default_exp')

    p = parser.parse_args()
    args = vars(p)
    namespace_dir = os.path.join(dir_path, 'data/exps', args['namespace'])
    os.makedirs(namespace_dir, exist_ok=True)
    args['namespace_dir'] = namespace_dir

    transform = ToTensor()
    imgs, ann_imgs = pull_training_data(args['training_dir'])
    dataset = WaterLevelDataset(imgs[:14], ann_imgs[:14], transform=transform)
    test_dataset = WaterLevelDataset(imgs[14:], ann_imgs[14:], transform=transform)

    model = train_model(args, dataset)
    test_outp_dir = os.path.join(args['namespace_dir'], 'test_figures')
    os.makedirs(test_outp_dir, exist_ok=True)
    generate_output(test_dataset, model, outdir=test_outp_dir, T=1e-2)
