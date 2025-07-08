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
    def __init__(self, X, Y, device):
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
    
training_data = dataset(x_train, y_train, device)
validation_data = dataset(x_val, y_val, device)
testing_data = dataset(x_test, y_test, device)


#data loader
'''
    Dataloader is an object that we can loop through it to train according to batches.
    When we start training, we loop through epochs, if you skip the vatch size it means that 
    the amount of training data in one batch is equal to the complete amount of training data,
    this method is not efficient and in most of the cases you need to train through using batches.

    When you create a dataloader, you define the batch size and enable the shuffle to randomize the data
    and then you can loop through it each epoch to train normally.
'''
train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=8, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=8, shuffle=True)

# build the model
HIDDEN_NEURONS = 10
class MyModel(nn.Module):
    #constructor
    def __init__(self):
        super(MyModel, self).__init__() #we use the same constructor in nn.Module
        #create layer
        '''
            a linear layer which represents the input and having the input size of 10 
            which is the number of columns of the input and the output of the number of hidden neurons, 
            next layer is the output layer which have the input of hidden neurons and 
            one output since we have a binary classification. 
            Finally we have the activation function which is the sigmoid.
        '''
        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.linear = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()

    # Define how the data flows inside the model
    def forward(self, x):
        # first, the input data x goes into the input layer
        x = self.input_layer(x)
        # then, the output of the input layer will go to the linear layer
        x = self.linear(x)

        x = self.sigmoid(x)
        return x
    
# tips: everything you create in the PyTorch needs to be moved into 'device'
model = MyModel().to(device)

'''
    1. Takes your model (model)
    2. Simulates a forward pass using a dummy input of shape (batch_size=1, X.shape[1])
       Here X.shape[1] is the size of the feature dimension of your data (i.e. number of input features).
    3. Prints out a layer-by-layer table showing:
       Each layer's name/type
       The output shape at that layer
       The number of parameters (weights & biases) in that layer
'''
# attention: you have to run model on CPU rather than GPU to call summary function
# because summary will use dummy data which locates on CPU.
summary(model.cpu(), (X.shape[1],))
model = MyModel().to(device)

#loss function (it's named criterion in most references)
criterion = nn.BCELoss() # BCELoss is the Binary Cross-Entropy for binary classification
optimizer = Adam(model.parameters(), lr=1e-3)

total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

epochs = 10

for epoch in range(epochs):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    for data in train_dataloader:
        inputs, labels = data
        # get the ouput from the model
        prediction = model(inputs).squeeze(1)

        #print(prediction.shape)            #the output is torch.Size([8, 1]) 就是有8 rows
        #print(prediction.squeeze(1).shape) #the output is torch.Size([8]). 就是有 8 columns
        
        batch_loss = criterion(prediction, labels)
        total_loss_train += batch_loss.item()

        #the output be like: tensor([False, True, False, True, False, False, True, True], device='mps')
        #because we want to know the accuracy, so sum
        #after sum, we will get "tensor(3, device=...)", which means we get 3 True out of 8
        #item will give you the only number -- 3
        #the batch's accuracy
        acc = ((prediction).round() == labels).sum().item()
        total_acc_train += acc

        #backpropagation
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    # this means you use the pytorch model not for training.
    # Just for testing. 
    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data
            prediction = model(inputs).squeeze(1)
            batch_loss = criterion(prediction, labels)
            total_loss_val += batch_loss.item()
            acc = ((prediction).round() == labels).sum().item()
            total_acc_val += acc

    # divide by 1000 to do normalization.
    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_validation_plot.append(round(total_loss_val/1000, 4))

    #calculate the precentage
    # __len__() give us the number of data set.
    total_acc_train_plot.append(round(total_acc_train/(training_data.__len__()) * 100, 4))
    total_acc_validation_plot.append(round(total_acc_val/(validation_data.__len__()) * 100, 4))

    print(f'''Epoch no. {epoch + 1} Train loss: {round(total_acc_train/1000, 4)} Train accuracy {round(total_acc_train/training_data.__len__() * 100, 4)}
            Validation Loss: {round(total_loss_val/1000, 4)} Validation Accuracy: {round(total_acc_val/validation_data.__len__() * 100, 4)}''')
    print('=' * 25)
    

    #do testing
    with torch.no_grad():
        total_loss_test = 0
        total_acc_test = 0
        for data in testing_dataloader:
            inputs, label = data
            prediction = model(inputs).squeeze(1)
            batch_loss_test = criterion(prediction, labels).item()
            total_loss_test += batch_loss_test
            acc = ((prediction).round() == labels).sum().item()
            total_acc_test += acc

    print("Accuracy: ", round(total_acc_test/testing_data.__len__() * 100, 4) )

#visualization
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_validation_plot, label='Validation Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0, 2])
axs[0].legend()

axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
axs[1].set_title('Training and Validation Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_ylim([0, 100])
axs[1].legend()

plt.tight_layout()

plt.show()


#inference

area = float(input("Area: "))/original_df['Area'].abs().max()
MajorAxisLength = float(input("Major Axis Length: "))/original_df['MajorAxisLength'].abs().max()
MinorAxisLength = float(input("Minor Axis Length: "))/original_df['MinorAxisLength'].abs().max()
Eccentricity = float(input("Eccentricity: "))/original_df['Eccentricity'].abs().max()
ConvexArea = float(input("Convex Area: "))/original_df['ConvexArea'].abs().max()
EquivDiameter = float(input("EquivDiameter: "))/original_df['EquivDiameter'].abs().max()
Extent = float(input("Extent: "))/original_df['Extent'].abs().max()
Perimeter = float(input("Perimeter: "))/original_df['Perimeter'].abs().max()
Roundness = float(input("Roundness: "))/original_df['Roundness'].abs().max()
AspectRation = float(input("AspectRation: "))/original_df['AspectRation'].abs().max()

my_inputs = [area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation]

print("="*20)

model_inputs = torch.Tensor(my_inputs).to(device)
prediction = (model(model_inputs))
print(prediction)
print("Class is: ", round(prediction.item()))
     