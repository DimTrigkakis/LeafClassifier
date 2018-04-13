import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os.path as path
import PIL
from PIL import Image
import random
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import torch.nn

import cv2

from torch.autograd import Variable
import matplotlib.pyplot as plt
import glob
import os
from xml.dom import minidom
from models import *

datapaths ={'annotated train':"C:/Users/Jack/Desktop/Bisque_Images/Test_folders"}
#datapaths ={'annotated train':"C:/Users/Jack/Desktop/Bisque_Images/Test_folders", "unsupervised":"C:/Users/Jack/Desktop/Bisque_Images/Unsupervised_folders"}
mydata = {key:[] for key in datapaths.keys()}
mydata_segmentation = {'all':[]}
imagefile_iter = {key:None for key in datapaths.keys()}
decoder_tags = ['ImageFileName','Leaf','LeafType','LeafShape','Leafbaseshape','Leaftipshape','Leafmargin','Leafvenation','Species','Contributor','BBox','restore']
leaf_targets = {'LeafType':["SIMPLE","COMPOUND","NONE"],'LeafShape':["ACEROSE","AWL-SHAPED","GLADIATE","HASTATE","CORDATE","DELTOID","LANCEOLATE","LINEAR","ELLIPTIC","ENSIFORM","LYRATE",
                   "OBCORDATE","FALCATE","FLABELLATE","OBDELTOID","OBELLIPTIC","OBLANCEOLATE","OBLONG","PERFOLIATE","QUADRATE","OBOVATE","ORBICULAR",
                   "RENIFORM","RHOMBIC","OVAL","OVATE","ROTUND","SAGITTATE","PANDURATE","PELTATE","SPATULATE","SUBULATE","NONE"],
                'Leafbaseshape':["AEQUILATERAL","ATTENUATE","AURICULATE","CORDATE","CUNEATE","HASTATE","OBLIQUE","ROUNDED","SAGITTATE","TRUNCATE","NONE"],
                'Leaftipshape': ["CIRROSE","CUSPIDATE","ACUMINATE","ACUTE","EMARGINATE","MUCRONATE","APICULATE","ARISTATE","MUCRONULATE","MUTICOUS","ARISTULATE","CAUDATE","OBCORDATE","OBTUSE","RETUSE","ROUNDED","SUBACUTE","TRUNCATE","NONE"]
                   ,'Leafmargin':["BIDENTATE","BIFID","DENTATE","DENTICULATE","BIPINNATIFID","BISERRATE","DIGITATE","DISSECTED","CLEFT","CRENATE","DIVIDED","ENTIRE","CRENULATE","CRISPED","EROSE","INCISED","INVOLUTE","LACERATE","PEDATE","PINNATIFID","LACINIATE","LOBED","PINNATILOBATE","PINNATISECT","LOBULATE","PALMATIFID","REPAND","REVOLUTE","PALMATISECT","PARTED","RUNCINATE","SERRATE","SERRULATE","SINUATE","TRIDENTATE","TRIFID","TRIPARTITE","TRIPINNATIFID","NONE"],'Leafvenation':["RETICULATE","PARALLEL","NONE"]}

for key in datapaths.keys():
    if key == "annotated train":
        fileforms = datapaths[key]+'/**_done/*.jpg'
    elif key == "unsupervised":
        fileforms = datapaths[key]+'/**/*.jpg'

    imagefile_iter[key] = glob.iglob(fileforms, recursive=True)

def decoder(image_generator):

    sample_lists = {key: [] for key in datapaths.keys()}
    for key in datapaths.keys():
        if key == "annotated train":
            for i, item in enumerate(image_generator[key]):
                imagefile = item
                xmlfile = item[:-4]+".xml"
                assert os.path.exists(imagefile)
                assert os.path.exists(xmlfile)
                sample = {"image":imagefile, "xml":xmlfile}
                sample_lists[key].append(sample)

        elif key == "unsupervised":
            for i, item in enumerate(image_generator[key]):
                if "done" not in item:
                    imagefile = item
                    xmlfile = item[:-4] + ".xml"
                    assert os.path.exists(imagefile)
                    assert os.path.exists(xmlfile)
                    sample = {"image":imagefile, "xml":xmlfile}
                    sample_lists[key].append(sample)

    return sample_lists

def full_decoder(image_generator):
    print("building generators")
    sample_lists = decoder(image_generator)
    errors = {key:0 for key in datapaths.keys()}
    for key in datapaths.keys():
        print("building full generator {}".format(key))
        if key == "annotated train":
            for sample in sample_lists[key]:
                imagefile = sample['image']
                xmlfile = sample['xml']

                try:
                    image = PIL.Image.open(imagefile)
                    xmldoc = minidom.parse(xmlfile)

                    propertylist = {key:None for key in decoder_tags}
                    for mytag in decoder_tags:
                        propertylist[mytag] = xmldoc.getElementsByTagName(mytag)[0].firstChild.nodeValue

                    datum = {"image":image, "info":propertylist}
                    mydata[key].append(datum)
                    mydata_segmentation['all'].append(datum)

                except:
                    errors[key] += 1

        elif key == "unsupervised":
            for sample in sample_lists[key]:
                imagefile = sample['image']
                xmlfile = sample['xml']

                try:
                    image = PIL.Image.open(imagefile)
                    xmldoc = minidom.parse(xmlfile)

                    if  xmldoc.getElementsByTagName('Type')[0].firstChild.nodeValue == "SheetAsBackground":
                        datum = {"image":image, "info":'unsupervised'}
                        mydata[key].append(datum)
                        mydata_segmentation['all'].append(datum)
                except:
                    errors[key] += 1

    for key in datapaths.keys():
        print("Created data for subset {} with {} elements with {} errors".format(key, len(mydata[key]), errors[key]))

full_decoder(imagefile_iter)

# The datasets have been created

# Two data-loaders: A) mstd calculated for cnns, B) for segmentation (all images only for segmentation)
class DataBuilderAnnotated(data.Dataset):

    def segmentation_transform(self, mean, std):

        t = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) # H x W resize
        return t

    def __init__(self):
        self.loader_segmentation = torch.utils.data.DataLoader(dataset=self, batch_size=32, shuffle=False)
        self.t = self.segmentation_transform(mean=[0.7446,0.7655,0.7067], std=[0.27776,0.24386,0.33867])

    def loader(self):
        return self.loader_segmentation

    def __getitem__(self, index):
        datum = mydata['annotated train'][index] # B x C x H x W

        info = []
        for tag in leaf_targets.keys():
            info.append(torch.LongTensor([int(leaf_targets[tag].index(datum["info"][tag]))]))
        return {"Image":self.t(datum["image"]),"Info":info}

    def __len__(self):
        return len(mydata['annotated train'])

class DataBuilderSegmentation(data.Dataset):

    def segmentation_transform(self, mean, std):

        t = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) # H x W resize

        return t

    def __init__(self):
        self.loader_segmentation = torch.utils.data.DataLoader(dataset=self, batch_size=32, shuffle=False)
        self.t = self.segmentation_transform(mean=[0.0,0.0,0.0], std=[1,1,1])

    def loader(self):
        return self.loader_segmentation

    def __getitem__(self, index):
        datum = mydata['annotated train'][index] # B x C x H x W
        return {"Image":self.t(datum["image"])}

    def __len__(self):
        return len(mydata['annotated train'])

dataloader_segmentation = DataBuilderSegmentation()
dataloader_annotated = DataBuilderAnnotated()

################# MSTD calculation for a subset of the dataset (annotated part)
'''
# Compute mstd

mean = torch.zeros(3)
total_batches = 0

for i, datum in enumerate(dataloader_annotated.loader()): # to ignore last batch correction, set batch size to 1
    tensor_mean = datum['Image']
    for dim in [0,1,1]:
        tensor_mean = torch.mean(tensor_mean, dim)

    mean += tensor_mean
    total_batches += 1

total_batches = 0
var = np.zeros(3)

for i, datum in enumerate(dataloader_annotated.loader()):
    total_batches += 1
    var += np.var(datum['Image'].numpy(), axis=(0,2,3))
'''

################# Segmentation

for bidx, datum in enumerate(dataloader_segmentation.loader()):
    for i in range(datum["Image"].size()[0]):

        # Saturated segmentation for this dataset (grayscale background vs plant)

        img = torch.mul(datum["Image"][i], 255).numpy().swapaxes(0,2).swapaxes(0,1).astype(np.uint8).copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1]

        th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        _, cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea)

        for cnt in cnts:
            continue

        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        #dst = cv2.bitwise_and(img, img, mask=mask)

        th2, threshed2 = cv2.threshold(hsv_saturation, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        _, cnts2, _ = cv2.findContours(threshed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = sorted(cnts2, key=cv2.contourArea)

        for cnt2 in cnts2:
            continue

        mask2 = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask2, [cnt2], -1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)
        dst2 = cv2.bitwise_and(img, img, mask=mask2)

        row = np.nonzero(np.maximum.reduce(np.maximum.reduce(dst2,0),1))
        row_x1, row_xn = np.min(row), np.max(row)
        col = np.nonzero(np.maximum.reduce(np.maximum.reduce(dst2, 2), 1))
        col_x1, col_xn = np.min(col), np.max(col)
        box = {'xa':row_x1, 'xb':row_xn, 'ya':col_x1, 'yb':col_xn}
        width = box['xb'] - box['xa']
        height = box['yb'] - box['ya']
        box['w'] = width
        box['h'] = height

        #cv2.rectangle(dst2, (box['xa'], box['yb']), (box['xb'], box['ya']), (255, 0, 0), 2)
        #cv2.rectangle(dst2, (box['xa']+3, box['yb']-3), (box['xb']-3, box['ya']+2*box['h']//3+3), (0, 255, 0), 2)
        #cv2.rectangle(dst2, (box['xa']+3, box['yb']-2*box['h']//3-3), (box['xb']-3, box['ya']+3), (0, 0, 255), 2)
        cv2.rectangle(img, (box['xa'], box['yb']), (box['xb'], box['ya']), (255, 0, 0), 2)
        cv2.rectangle(img, (box['xa']+3, box['yb']-3), (box['xb']-3, box['ya']+2*box['h']//3+3), (0, 255, 0), 2)
        cv2.rectangle(img, (box['xa']+3, box['yb']-2*box['h']//3-3), (box['xb']-3, box['ya']+3), (0, 0, 255), 2)

        cv2.imshow("dst.png", img)
        cv2.waitKey()

        ## Save it
    break
# Return bounding boxes on any leaf image given as input
# Sample the first image through the segmentation builder
# segment it into parts
# create the tensors associated with the bounding boxes, using a specific function
# classify parts independently

exit()
################# Naive classification

mymodel = resnet18(True)
mymodel.reform()
mymodel.cuda()
optimizer = optim.Adam(mymodel.parameters(), weight_decay=0.0, lr= 1e-4)
criterion = torch.nn.CrossEntropyLoss()
best_preds = [[0 for j in range(len(leaf_targets[i]))] for i in leaf_targets.keys()]

samples = 0
for bidx, datum in enumerate(dataloader_annotated.loader()):
    target = datum["Info"]
    bs = datum["Image"].size()[0]
    for i in range(bs):
        for tag in range(6):
            best_preds[tag][target[tag][i].cpu().numpy()[0]] += 1
    samples += bs

for i in range(len(leaf_targets.keys())):
    max = 0
    for j in range(len(best_preds[i])):
        if best_preds[i][j]/samples > max:
            max = best_preds[i][j]/samples
    print(best_preds[i][0]/samples)

for epoch in range(10):
    for bidx, datum in enumerate(dataloader_annotated.loader()):
        image = datum["Image"]
        output = mymodel(Variable(image).cuda())
        L_class = 0
        # build targets based on datum info
        target = datum["Info"]

        correct = []
        for i in range(6):
            correct.append(0)
            L_class += criterion(output[i],Variable(target[i].squeeze()).cuda())
            for k in range(image.size()[0]):
                if target[i][k][0] == np.argmax(output[i].data.cpu().numpy(), axis=1)[k]:
                    correct[i] += 1

        for i in range(6):
            mykeys = list(leaf_targets.keys())
            print('Accuracy for category {}'.format(mykeys[i]),correct[i]/image.size()[0])
        optimizer.zero_grad()
        L_class.backward()
        optimizer.step()

        if epoch != 9 and bidx > 14:
            break

        print(bidx)


    #print(L_class.data.cpu().numpy())






