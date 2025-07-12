#The dataset is divided into 3 categories - cat, dog and wildlife


import torch #mian torch library
from torch import nn #nn - which has the layers and the loss function
from torch.optim import Adam #import optimizer Adam used in training
from torchvision.transforms import transforms #for pre-processing images
from sklearn.preprocessing import LabelEncoder #convert the output data from strings to integers 
from torch.utils.data import Dataset, DataLoader #create custom data sets, and dataloader is the object to loop for batches
import matplotlib.pyplot as plt #for plotting
from PIL import Image #to read the images
import pandas as pd #for data pre-processing
import numpy as np #for data pre-processing
import os #read data from the directory

#PyTorch does not automatically detect the GPU, so you have to specify it
#(TensorFlow can detect GPU automatically)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#First of all, we need to read all the data
#The original dataset only has train and validation, we need to split the dataset into train, validation and test
#Think about the directories as layers
base_dir = "/Users/yangtianjiao/projects/Python_Projects/NeuralNetworks_Study/animal_faces"
image_path = []
labels = []
#this i means 'train' folder and 'val' folder
#attention: the MacOS will create .xxx - the hide files.
for split in os.listdir(base_dir):
    split_path = os.path.join(base_dir, split)
    if not os.path.isdir(split_path):
        continue

    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            if os.path.isfile(file_path):
                image_path.append(file_path)
                labels.append(label)

data_df = pd.DataFrame(zip(image_path, labels), columns=["image_path", "labels"])
print(data_df["labels"].unique()) #three: ['wild', 'dog', 'cat']

train = data_df.sample(frac=0.7) #we take 70% amount of the original data frame as training data
test = data_df.drop(train.index) #the pending 30% will be used as testing data
val = test.sample(frac=0.5) #take the validation data set from the testing data set
test = test.drop(val.index) #remember to update the testing dataset
print(train.shape)
print(val.shape)
print(test.shape)