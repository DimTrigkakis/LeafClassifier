import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import glob
import PIL
from PIL import Image
from torchvision.utils import save_image as save
import random
from models import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
from enum import Enum

class SetType(Enum):
    TRAIN = 0
    TEST = 1
    GLOBAL = 2

class DataBuilder(data.Dataset):

    @staticmethod
    def pair_transform():
        normalize = transforms.Normalize(
            mean=[0.549, 0.570, 0.469],
            std=[0.008, 0.0085, 0.0135]
        )

        normal_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize([224, 224]),
            #transforms.RandomCrop(224, padding=0),
            transforms.ToTensor(),  # C x H x W
            normalize
        ])

        return normal_transform

    def __init__(self, configuration):
        self.mode = SetType.TRAIN
        self.pt = self.pair_transform()
        os.environ["CUDA_VISIBLE_DEVICES"]= configuration['CUDA']
        self.configuration = configuration
        if configuration['sampler']['type'] is not None:
            self.trainloader = torch.utils.data.DataLoader(dataset=self, batch_size=configuration['sampler'][SetType.TRAIN]['bs'],
                                                        num_workers=12, sampler=configuration['sampler']['type'])
        else:
            self.trainloader = torch.utils.data.DataLoader(dataset=self, batch_size=configuration['sampler'][SetType.TRAIN]['bs'],
                                                        num_workers=12, shuffle=configuration['sampler'][SetType.TRAIN]['shuffle'])

        self.testloader = torch.utils.data.DataLoader(dataset=self, batch_size=configuration['sampler'][SetType.TEST]['bs'],
                                                    shuffle=configuration['sampler'][SetType.TEST]['shuffle'],
                                                    num_workers=12)

    def loader(self, mode=SetType.TRAIN):
        self.mode = mode
        if mode == SetType.TRAIN:
            return self.trainloader
        else:
            return self.testloader

    def __getitem__(self, index):
        datum = self.configuration['data'][self.mode][index]
        return self.configuration['decoder'](datum)

    def __len__(self):
        return len(self.configuration['data'][self.mode])


configuration = {'data': {SetType.TRAIN: [], SetType.TEST: []}, "CUDA":"1", "mode": SetType.TRAIN, 'decoder': None,
                    'length': -1, 'sampler': {SetType.TRAIN: None, SetType.TEST: None}, 'sampling_weights': None, 'subset': {SetType.TRAIN: None, SetType.TEST: None}}

########## Globbers

datapaths = {SetType.TRAIN:["/scratch/Jack/datasets/RA_datasets/ImageClef/training_data","/scratch/Jack/datasets/RA_datasets/Pankaj/"],
                    SetType.TEST:["/scratch/Jack/datasets/RA_datasets/ImageClef/testing_data"]}
dbglob = {SetType.TRAIN: [], SetType.TEST: []}
for setType in [SetType.TRAIN, SetType.TEST]:
    for path in datapaths[setType]:
        dbglob[setType] += glob.glob(path + "/**/*.jpg", recursive=True)

classes = ["leaf","fruit","flower","stem","entire"]
class_numbers = {"leaf":0,"fruit":0,"flower":0,"stem":0,"entire":0}
count_train = 0

for setType in [SetType.TRAIN, SetType.TEST]:
    for item in dbglob[setType]:
        #if "entire" not in item:
        label = -1
        for i, c in enumerate(classes):
            if c in item:
                label = i
        if label == -1:
            continue
        class_numbers[classes[label]] += 1
        if SetType.TRAIN == setType:
            count_train += 1
        configuration["data"][setType].append(item)

################ Subset formation

configuration['subset'][SetType.TRAIN] = 1.0
configuration['subset'][SetType.TEST] = 1.0

def specific_subset_selection(mode="train"):
    index = 0
    count_max = len(configuration['data'][mode])
    for i in range(count_max):
        if i % int(1.0 / configuration['subset'][mode]) != 0:
            configuration['data'][mode].pop(index)
        else:
            index += 1

for setType in [SetType.TRAIN, SetType.TEST]:
    specific_subset_selection(setType)

############### Decoder

pt = DataBuilder.pair_transform()
def mapping_decoder(datum):
    image = pt(PIL.Image.open(datum))
    label = -1
    for i, c in enumerate(classes):
        if c in datum:
            label = i
    assert label != -1
    sample = {"Image": image, "Label": int(label)}

    return sample

############### Sampler

weight_per_class = [0.] * len(classes)

weight = [0] * len(configuration["data"][SetType.TRAIN])
weights = torch.DoubleTensor(weight)
N = float(count_train)
for i in range(len(classes)):
        weight_per_class[i] = N/float(class_numbers[classes[i]])

for idx, item in enumerate(configuration["data"][SetType.TRAIN]):
    for i, c in enumerate(classes):
        if c in item:
            label = i
    weight[idx] = weight_per_class[label]

sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

############ Model

model = resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048,5)
model.cuda()

########### Configuration

# configuration['sampler']['type'] = None
configuration['sampler']['type'] = sampler
configuration['model'] = model
configuration['decoder'] = mapping_decoder
configuration['sampler'][SetType.TRAIN] = {'bs': 64, 'shuffle': True}
configuration['sampler'][SetType.TEST] = {'bs': 64, 'shuffle': False}
configuration['optimizer'] = optim.Adam(configuration['model'].parameters(), lr=0.0001,weight_decay=0.0008)

#### DataBuilder

db = DataBuilder(configuration)

# Mean calculation
'''
mean = torch.zeros((3))

for i, datum in enumerate(db.trainloader):
    image_tensor = datum["Image"]
    mean_local = image_tensor
    for dim in [0,1,1]:
        mean_local = torch.mean(mean_local,dim=dim)
    mean = torch.add(mean, mean_local)
    print(mean/(i+1))
    
std = torch.zeros((3))

# Std calculation after normalizing with the mean
for i, datum in enumerate(db.trainloader):
    image_tensor = datum["Image"]
    std_local = image_tensor
    for dim in [0,1,1]:
        std_local = torch.std(std_local,dim=dim)
    std = torch.add(std, std_local)
    print(std/(i+1))
'''

####

def train(epoch):
    model.train()
    for batch_idx, datum in enumerate(db.loader(SetType.TRAIN)):
        data, target = Variable(datum["Image"].cuda()), Variable(datum["Label"].cuda())
        configuration['optimizer'].zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        configuration['optimizer'].step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, int(batch_idx * len(data)/target.size()[0]), len(db.trainloader),
                100. * batch_idx/ len(db.trainloader), loss.data[0]))


def test(dtt):
    conf_matrix = torch.zeros((5,5))
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, datum in enumerate(db.loader(dtt)):
        data, target = datum["Image"].cuda(), datum["Label"].cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, size_average=False).data[0] # sum up batch loss
        pred = F.log_softmax(output, dim=1).data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total += target.size()[0]
        for i in range(target.size()[0]):
            conf_matrix[int(pred[i].cpu().numpy())][int(target[i].data.cpu().numpy())] += 1

        print(batch_idx, len(db.loader(dtt)), correct)
    print(conf_matrix)
    test_loss /= len(db.testloader)*db.configuration['sampler'][dtt]["bs"]
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))

for i in range(5):

    torch.save(model.state_dict(),"./model_e{}".format(i)+".pth")
    train(i)
    test(SetType.TRAIN)
    test(SetType.TEST)
    

# Pankaj 656 to 679 batches
# from 91% test acc to 91% in 3 epochs (100% training acc)
# pytorch data augmentation (remove because of entire)