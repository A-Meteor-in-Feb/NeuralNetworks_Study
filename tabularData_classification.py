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


