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
        if 'target' in sample:
            input, target = sample['input'], sample['target']
            target = (target !=0).astype('int')
        else:
            input = sample['input']
        
        input = input.transpose((2, 0, 1))
        input = input /255.0

        if 'target' in sample:

            return {'input': torch.from_numpy(input).float(),
                'target': torch.from_numpy(target).float()}
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
    def __init__(self,num_channels,filter_size=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, num_channels, filter_size, padding='same')
        self.conv2 = nn.Conv2d(num_channels, num_channels, filter_size, padding='same')
        self.conv3 = nn.Conv2d(num_channels,1,1)
        self.deploy_temperature = -1
        self.deploy_flag = False

    def set_deploy(self,deploy_flag,deploy_temperature):
        # deploy flag to be set for tracing and onnxruntime. 
        self.deploy_flag = deploy_flag
        if self.deploy_flag:
            self.deploy_temperature = deploy_temperature

    def forward(self, x):
        original_dim = x.dim()
        if original_dim==3:
          x = x.unsqueeze(0)
        x1 = F.relu(self.conv1(x))
        #print(x1.shape)
        x2 = self.conv3(F.relu(self.conv2(x1)))
        #print(x2.shape)
        x3 = x2.squeeze(1)
        #print(x3.shape)

        if self.deploy_flag is True and self.deploy_temperature >0:
            x3 = F.softmax(x3/self.deploy_temperature, dim = 1)

        if original_dim==3:
          x3 = x3.squeeze(0)

        return x3

def get_annotation_mask(annotated):
  annotation_mask = (annotated[:,:,0]/(annotated[:,:,1]+1e-10)>1.8) * (annotated[:,:,0]/(1e-10+annotated[:,:,2])>1.8)
  annotation_mask[0:50] = 0 
  annotation_mask[150:] =0
  return annotation_mask

def pull_training_data(DATADIR):
    filenames = [os.path.join(DATADIR,_) for _ in os.listdir(DATADIR)
           if os.path.isfile(os.path.join(DATADIR,_)) and _.endswith('.bmp')]
           
    annotated_filenames = [_.replace('.bmp',' copy.jpeg') for _ in filenames]
    imgs, ann_imgs = [],[]
    for f_,a_f_ in zip(filenames,annotated_filenames):
        original = Image.open(f_)
        original = np.asarray(original)
        original = original[:,:,:3]

        annotated = Image.open(a_f_)
        annotated = np.asarray(annotated)
        
        annotation_mask = get_annotation_mask(annotated)

        imgs.append(original)
        ann_imgs.append(annotation_mask)


        ann_image= np.array(original)
        ann_image[annotation_mask] = np.array([255,0,0])
    # plt.imshow(ann_image)
    # plt.show()

    return imgs,ann_imgs

def train_model(args,dataset):

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                            shuffle=True)

    #model = WTNet(num_channels = 15, filter_size = 15)
    model = WT3Net(num_channels = 10, filter_size = 4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    max_epochs = args['max_epochs']

    for epoch_idx in range(max_epochs):
        loss_list = []
        t_list,p_list =[],[]
        for dta in trainloader:
            inputs = dta['input']
            targets = dta['target']
            norm_targets = targets + 1e-9
            norm_targets = norm_targets / torch.sum(norm_targets,axis=1,keepdims=True)
            outp = model(inputs)
            losses = criterion(outp,norm_targets)

            loss = losses.mean()
            loss_list.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        if (epoch_idx+1) %5 ==0:
            print(f'Epoch {epoch_idx+1}/{max_epochs} Loss: {np.mean(loss_list)}')
            #print(np.mean(np.array(t_list)==np.array(p_list)))
    
    model_path = os.path.join(args['namespace_dir'],'model.pt')
    torch.save(model.state_dict(), model_path)


    grayscale_outdir = os.path.join(args['namespace_dir'],'grayscale_out')
    os.makedirs(grayscale_outdir)
    for idx in range(dataset.__len__()):
        dta = dataset[idx]
        inputs = dta['input']
        targets = dta['target']
        img = inputs.cpu().detach().numpy().transpose((1,2,0))
        outp = F.softmax(model(inputs),dim = 0).cpu().detach().numpy()
        # plt.imshow(outp,cmap='gray')
        plt.imsave(os.path.join(grayscale_outdir,f'train_{idx}.jpg'),
            outp,
            cmap='gray')

    return model

def generate_output(dataset,model,outdir,T = 1e-2):

    for idx in range(dataset.__len__()):
        dta = dataset[idx]
        inputs = dta['input'] 
        targets = dta['target']

        img = inputs.cpu().detach().numpy().transpose((1,2,0))
        outp = F.softmax((model(inputs)/T),dim = 0).cpu().detach().numpy()


        img[:,:,0] +=outp
        img[:,:,0] = img[:,:,0]/img[:,:,0].max()
        # plt.imshow(img)
        # plt.show()
        plt.imsave(os.path.join(outdir,f'img_{idx}.jpg'),img)
        outp = F.softmax(model(inputs),dim = 0).cpu().detach().numpy()

        plt.imsave(os.path.join(outdir,f'img_gray_{idx}.jpg'),outp,cmap='gray')
        



if __name__=="__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir",type=Path,default=os.path.join(dir_path,"data/annotated_bmp"),help="Path to the training data directory",)
    parser.add_argument('--max_epochs',type=str,default=50)
    parser.add_argument('-n','--namespace',type=str,default='default_exp')

    p = parser.parse_args()
    args = vars(p)
    namespace_dir = os.path.join(dir_path,'data/exps',args['namespace'])
    if os.path.exists(namespace_dir):
        import shutil
        shutil.rmtree(namespace_dir, ignore_errors=True)
    os.makedirs(namespace_dir)
    args['namespace_dir'] = namespace_dir
    print(f'Arguments :\n{args}')


    transform = ToTensor()
    imgs,ann_imgs = pull_training_data(args['training_dir'])
    dataset = WaterLevelDataset(imgs[:14],ann_imgs[:14],transform = transform)
    test_dataset = WaterLevelDataset(imgs[14:],ann_imgs[14:],transform = transform)


    model = train_model(args,dataset)

    test_outp_dir =os.path.join(args['namespace_dir'],'test_figures')
    os.makedirs(test_outp_dir)
    generate_output(test_dataset,model, outdir= test_outp_dir,T = 1e-2)
