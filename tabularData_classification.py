import torch # main library of pytorch
import torch.nn as nn #nn stands for neural network, any layer will be in torch.nn, including cnn, lstm, rnn and so on.
from torch.optim import Adam #import Adam optimizer
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary #to print how the model architecture will be at the end
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt #to plot the training progress
import pandas as pd
import numpy as np

#PyTorch does not automatically detect the GPU, so you have to specify it
#(TensorFlow can detect GPU automatically)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device: {device}")

# pd.read_csv(" ") read from csv file
# load it as DataFrame object 
data_df = pd.read_csv("riceClassification.csv")
# data_df.head() from pandas
# return the first few rows of data to check the data's structure and so on.
print(data_df.head()) 

# do a little bit pre-processing
# 1) drop the missing values
data_df.dropna(inplace=True)
# 2) drop the id column, because id has no use.
data_df.drop(['id'], axis=1, inplace=True) #axis = 1 means it is a column

print(data_df.shape) # attention - no '()' after shape

# Then, get the outputs' posibilities, what kinds of outputs are
print(data_df["Class"].unique())
# We also can get the frequencies of the different types
print(data_df["Class"].value_counts())

# Then, another part of preprocessing - normalization.
'''
    - Why we want to do normalization?
    - The actual values are extremely huge, this is not good.
      Usually, we want the values to be samll because the weigths of the neural network
      are reletively small compared to the actual values.
      Also, when the weights are multiplied by the inputs, throughout the different layers 
      of neural network, these values are becoming very huge, which means that it will be 
      computationally expensive. Also, when you calculate the loss function, it may round up the values.
'''
original_df = data_df.copy() # because we will need the old data values for the inference.
# normalization - divide each value by the max value
for column in data_df.columns:
    data_df[column] = data_df[column]/data_df[column].abs().max()

X = np.array(data_df.iloc[:, :-1]) # take all the rows, all the columns except the last one.
Y = np.array(data_df.iloc[:, -1])  # take all the rows and the last column.
# split the training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
print(x_train.shape) #(12729, 10)
print(x_test.shape)  #(2728, 10)
print(x_val.shape)   #(2728, 10)

'''
    - How PyTorch understands the data, how PyTorch sees the data?
    - It doesn't see it as the pandas, now the data is a pandas or a numpy array,
      now, we want a PyTorch dataset object. 
      So, we need to convert our data into PyTorch dataset object.
'''
# some useful functions are already defined in the pytorch, 
# we need to modify to macth our dataset and to be suitable to our dataset.
class dataset(Dataset):
    # constructor
    def __init__(self, X, Y):
        # The input velue (maybe pandas, numpy arry or normal list) is converted into torch.tensors
        # move the converted data to the device (GPU).
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype=torch.float32).to(device)

    # return the shape of input data
    def __len__(self):
        return len(self.X) 
    
    # get the specific item according to the index
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
training_data = dataset[x_train, y_train]
validation_data = dataset[x_val, y_val]
testing_data = dataset[x_test, y_test]


#data loader
