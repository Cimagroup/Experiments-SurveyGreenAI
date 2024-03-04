###################################
# DATA REDUCTION EXPERIMENTS
# Collision Dataset
###################################
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from codecarbon import OfflineEmissionsTracker
from data_reduction.statistic import srs_selection, prd_selection
from data_reduction.geometric import clc_selection, mms_selection, des_selection
from data_reduction.ranking import phl_selection, nrmd_selection
from data_reduction.wrapper import fes_selection
from data_reduction.representativeness import find_epsilon
###################################

###################################
# 1.
# Load, scale and shuffle the dataset
###################################

#Loading
print("Loading dataset...")
df = pd.read_excel('collision.xlsx')
print("Dataset loaded.")
X_initial = df.drop(columns=['N','m','collision']).to_numpy()
y_initial = df['collision'].to_numpy()
#Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_initial)
y_scaled = y_initial
#Shuffling
np.random.seed(2024)
random_indices = np.random.permutation(len(y_scaled))
X_shuffled = X_scaled[random_indices]
y_shuffled = y_scaled[random_indices]
#Number of features and classes
n_f = X_shuffled.shape[1]
n_c = len(np.unique(y_shuffled))

###################################
# 2.
# Define the arguments object
###################################

parser = argparse.ArgumentParser(description='Arguments for the experiment')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    metavar='LR',
    help='Learning Rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--dropout_prob',
    type=float,
    default=0.33,
    metavar='M',
    help='Dropout probability (default: 0.33)')
parser.add_argument(
    '--total_epochs',
    type=int,
    default=200,
    metavar='N',
    help='number of epochs to train (default: 200)')
parser.add_argument(
    '--initial_epochs',
    type=int,
    default=100,
    metavar='N',
    help='number of epochs to train before reduction (default: 50)')
parser.add_argument(
    '--n_iter',
    type=int,
    default=10,
    metavar='N_iter',
    help='number of iterations of the experiment (default: 10)')
parser.add_argument(
    '--test_size',
    type=float,
    default=0.25,
    metavar='test_size',
    help='Proportional size of the test dataset (default: 0.25)')
parser.add_argument(
    '--device',
    type=str,
    default='cpu',
    help='Device to do the computations. Can be cou or cuda (default: cpu)')
parser.add_argument(
    '--country',
    type=str,
    default='ESP',
    help='ISO code of the country where the computations are being made (default: ESP). This is relevant to measure the carbon emissions with carboncode. See the ISO code of your country in: https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes')
parser.add_argument(
    '--filename',
    type=str,
    default='stats.pkl',
    help='Name of the .pkl file where the statistics dictionary should be saved.')

args = parser.parse_args([
    '--learning_rate','0.001',
    '--momentum','0.5',
    '--batch_size','1024',
    '--dropout_prob', '0.50',
    '--total_epochs','600',
    '--initial_epochs','200',
    '--n_iter','10',
    '--test_size','0.25',
    '--device', 'cuda',
    '--country', 'ESP',
    '--filename', 'collision_stats.pkl'
])

###################################
# 3.
# Define the neural arquitecture
###################################

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, dropout_prob):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        
        # Define the layers
        self.layer_1 = nn.Linear(input_size, 50)
        self.layer_2 = nn.Linear(50,45)
        self.layer_3 = nn.Linear(45,40)
        self.layer_4 = nn.Linear(40,35)
        self.layer_5 = nn.Linear(35,30)
        self.layer_6 = nn.Linear(30,25)
        self.layer_7 = nn.Linear(25,20)
        self.layer_8 = nn.Linear(20,15)
        self.layer_9 = nn.Linear(15,10)
        self.output_layer = nn.Linear(10, 1)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.layer_1(x)))
        x = self.dropout(torch.relu(self.layer_2(x)))
        x = self.dropout(torch.relu(self.layer_3(x)))
        x = self.dropout(torch.relu(self.layer_4(x)))
        x = self.dropout(torch.relu(self.layer_5(x)))
        x = self.dropout(torch.relu(self.layer_6(x)))
        x = self.dropout(torch.relu(self.layer_7(x)))
        x = self.dropout(torch.relu(self.layer_8(x)))
        x = self.dropout(torch.relu(self.layer_9(x)))
        x = self.output_layer(x)
        return x

###################################
# 4.
# Create the statistics dictionary
###################################

all_methods = ['SRS',
               'CLC',
               'MMS',
               'DES',
               'NRMD'
              ]
percentages = [0.1,
               0.2,
               0.3,
               0.4,
               0.5,
               0.6,
               0.7,
               0.8,
               0.9
               ]
metrics = ['time',
           'carbon',
           'epsilon',
           'acc',  
           'pre_0',
           'pre_1',
           'pre_avg',
           'rec_0',
           'rec_1',
           'rec_avg',
           'f1_0',
           'f1_1',
           'f1_avg'
          ]

stats = {}
for iter in range(args.n_iter):
    stats[iter] = {}
    for method_key in all_methods + ['FES']:
        stats[iter][method_key] = {}
        for percentage_key in percentages + [1.0]:
            stats[iter][method_key][percentage_key] = {}
            for metric_key in metrics:
                stats[iter][method_key][percentage_key][metric_key] = None

###################################
# 5.
# Functions to train the network
###################################

def create_new_model(n_f,args):
    model = NeuralNetwork(input_size=n_f,dropout_prob=args.dropout_prob)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    return model.to(args.device), criterion, optimizer

def tensorize(X,y,args):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor.to(args.device), y_tensor.to(args.device)

def save_stats(stats,args):
    with open(args.filename, "wb") as f:
        pickle.dump(stats, f)

def train_step(train_loader, model, args, criterion, optimizer):
    model = model.to(args.device)
    model.train() 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad() 
        output = model(data).view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def train_model(X,y,model,criterion,optimizer,args):
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    for i in range(args.total_epochs):
        print(f"\rEpoch: {i}", end='', flush=True)
        train_step(train_loader, model, args, criterion, optimizer)

def forgetting_step(model, current_accuracy, forgetting_events, X, y, args):
    model = model.to(args.device)
    model.eval()
    n_y = len(y)
    batch_size = args.batch_size
    with torch.no_grad():
        for i in range(0, int(n_y/batch_size)+1):
            batch_X = X[i*batch_size:i*batch_size+batch_size].to(args.device)
            batch_y = y[i*batch_size:i*batch_size+batch_size].to(args.device)
            predicted = predict(batch_X, model, args)
            correct = (predicted == batch_y).tolist()
            for j in range(len(correct)):
                indice = i * batch_size + j
                if indice > n_y:
                    continue
                forgetting_events[indice] += 1 if current_accuracy[indice] > correct[j] else 0
                current_accuracy[indice] = correct[j]

def train_fes(X,y,model,criterion,optimizer,args,p):
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    n_y = len(y)
    current_accuracy = np.zeros(n_y, dtype=np.int32)
    forgetting_events = np.zeros(n_y, dtype=np.int32)
    print("\n Epochs before reduction:")
    for i in range(args.initial_epochs):
        print(f"\rEpoch {i}", end='', flush=True)
        train_step(train_loader, model, args, criterion, optimizer)
        forgetting_step(model, current_accuracy, forgetting_events, X, y, args)
    X_red, y_red = fes_selection(y.to('cpu'),current_accuracy,forgetting_events,p,args.initial_epochs,X.to('cpu'))
    train_dataset_red = TensorDataset(X_red, y_red)
    train_loader_red = DataLoader(train_dataset_red, batch_size=args.batch_size, shuffle=True)
    print("\n Epochs after reduction:")
    for i in range(args.initial_epochs,args.total_epochs):
        print(f"\rEpoch {i}", end='', flush=True)
        train_step(train_loader_red, model, args, criterion, optimizer)
    return X_red, y_red

def predict(X_test_tensor, model, args):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.to(args.device))
    return torch.sigmoid(outputs).round().view(-1)
        

###################################
# 6.
# Function to comprise all 
# reduction methods
###################################

def reduce(X,y,perc,method):
    if method == 'SRS':
        X_red, y_red = srs_selection(X,y,perc)
    if method == 'CLC':
        X_red, y_red = clc_selection(X,y,perc)
    if method == 'MMS':
        X_red, y_red = mms_selection(X,y,perc)
    if method == 'DES':
        X_red, y_red = des_selection(X,y,perc,0.5*perc)
    if method == 'NRMD':
         X_red, y_red = nrmd_selection(X,y,perc,'SVD_python')
    return X_red, y_red

###################################
# 7.
# Functions to perform the
# experiment steps
###################################

# Step 1: Train the model with the total dataset (p=1.0)
def exp_step_1(X_train,y_train,X_test_tensor,y_test_tensor,args,stats,iter):
    print('\n Iteration ',iter)
    print("p = 1.0")
    #Set the model, criterion and optimizer
    model, criterion, optimizer = create_new_model(n_f,args)
    #Start the timer and the OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(country_iso_code=args.country,log_level="ERROR")
    tracker.start()
    start_time = time.time()
    #Reduce the dataset (no reduction for p=1.0)
    X_train_tensor, y_train_tensor = tensorize(X_train, y_train,args)
    #Train the model
    train_model(X_train_tensor,y_train_tensor,model,criterion,optimizer,args)
    #Stop the timer and the OfflineEmissionsTracker
    end_time = time.time()
    emission: float = tracker.stop()
    total_time = end_time - start_time
    #Test the model performance
    predicted = predict(X_test_tensor, model, args)
    cl_rep = classification_report(y_test_tensor.to('cpu'),predicted.to('cpu'), output_dict=True, zero_division = np.nan)
    #Save the results
    for m in all_methods + ['FES']:
        stats[iter][m][1.0]['time']=total_time
        stats[iter][m][1.0]['carbon']=emission
        stats[iter][m][1.0]['epsilon']=0
        stats[iter][m][1.0]['acc']=cl_rep['accuracy']
        stats[iter][m][1.0]['pre_0']=cl_rep['0.0']['precision']
        stats[iter][m][1.0]['rec_0']=cl_rep['0.0']['recall']
        stats[iter][m][1.0]['f1_0']=cl_rep['0.0']['f1-score']
        stats[iter][m][1.0]['pre_1']=cl_rep['1.0']['precision']
        stats[iter][m][1.0]['rec_1']=cl_rep['1.0']['recall']
        stats[iter][m][1.0]['f1_1']=cl_rep['1.0']['f1-score']
        stats[iter][m][1.0]['pre_avg']=cl_rep['macro avg']['precision']
        stats[iter][m][1.0]['rec_avg']=cl_rep['macro avg']['recall']
        stats[iter][m][1.0]['f1_avg']=cl_rep['macro avg']['f1-score']
        save_stats(stats,args)

# Step 2: Train the model with the reduced datasets (p in percentages)
def exp_step_mp(X_train,y_train,X_test_tensor,y_test_tensor,args,stats,iter):
    n_f = X_train.shape[1]
    n_c = len(np.unique(y_train))
    for m in all_methods:
        for p in percentages:
            print('\n Iteration ',iter)
            print("method =",m,"p =",p)
            #Set the model, criterion and optimizer
            model, criterion, optimizer = create_new_model(n_f,args)
            #Start the timer and the OfflineEmissionsTracker
            tracker = OfflineEmissionsTracker(country_iso_code=args.country,log_level="ERROR")
            tracker.start()
            start_time = time.time()
            #Reduce the dataset
            X_red, y_red = reduce(X_train, y_train, p, m)
            X_red_tensor, y_red_tensor = tensorize(X_red, y_red, args)
            #Train the model
            train_model(X_red_tensor,y_red_tensor,model,criterion,optimizer,args)
            #Stop the timer and the OfflineEmissionsTracker
            end_time = time.time()
            emission: float = tracker.stop()
            total_time = end_time - start_time
            #Test the model performance
            predicted = predict(X_test_tensor, model, args)
            cl_rep = classification_report(y_test_tensor.to('cpu'), predicted.to('cpu'), output_dict=True, zero_division = np.nan)
            #Save the results
            stats[iter][m][p]['time']=total_time
            stats[iter][m][p]['carbon']=emission
            stats[iter][m][p]['epsilon']=find_epsilon(X_train,y_train,X_red,y_red)
            stats[iter][m][p]['acc']=cl_rep['accuracy']
            stats[iter][m][p]['pre_0']=cl_rep['0.0']['precision']
            stats[iter][m][p]['rec_0']=cl_rep['0.0']['recall']
            stats[iter][m][p]['f1_0']=cl_rep['0.0']['f1-score']
            stats[iter][m][p]['pre_1']=cl_rep['1.0']['precision']
            stats[iter][m][p]['rec_1']=cl_rep['1.0']['recall']
            stats[iter][m][p]['f1_1']=cl_rep['1.0']['f1-score']
            stats[iter][m][p]['pre_avg']=cl_rep['macro avg']['precision']
            stats[iter][m][p]['rec_avg']=cl_rep['macro avg']['recall']
            stats[iter][m][p]['f1_avg']=cl_rep['macro avg']['f1-score']
            save_stats(stats,args)

# Step 3: Train the model with the datasets reduced with FES
def exp_step_fes(X_train,y_train,X_test_tensor,y_test_tensor,args,stats,iter):
    n_f = X_train.shape[1]
    n_c = len(np.unique(y_train))
    for p in percentages: 
        print('\n Iteration ',iter)
        print("method = FES","p =",p)
        #Set the model, criterion and optimizer
        model, criterion, optimizer = create_new_model(n_f,args)
        #Start the timer and the OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code=args.country,log_level="ERROR")
        tracker.start()
        start_time = time.time()
        #Train and reduce the model
        X_train_tensor, y_train_tensor = tensorize(X_train, y_train,args)
        X_red_tensor, y_red_tensor = train_fes(X_train_tensor,y_train_tensor,model,criterion,optimizer,args,p)
        #Stop the timer and the OfflineEmissionsTracker
        end_time = time.time()
        emission: float = tracker.stop()
        total_time = end_time - start_time
        #Test the model performance
        predicted = predict(X_test_tensor, model, args)
        cl_rep = classification_report(y_test_tensor.to('cpu'), predicted.to('cpu'), output_dict=True, zero_division = np.nan)
        #Save the results
        stats[iter]['FES'][p]['time']=total_time
        stats[iter]['FES'][p]['carbon']=emission
        stats[iter]['FES'][p]['epsilon'] = find_epsilon(X_train_tensor.to('cpu'),y_train_tensor.to('cpu'), X_red_tensor.to('cpu'),y_red_tensor.to('cpu')) 
        stats[iter]['FES'][p]['acc']=cl_rep['accuracy']
        stats[iter]['FES'][p]['pre_0']=cl_rep['0.0']['precision']
        stats[iter]['FES'][p]['rec_0']=cl_rep['0.0']['recall']
        stats[iter]['FES'][p]['f1_0']=cl_rep['0.0']['f1-score']
        stats[iter]['FES'][p]['pre_1']=cl_rep['1.0']['precision']
        stats[iter]['FES'][p]['rec_1']=cl_rep['1.0']['recall']
        stats[iter]['FES'][p]['f1_1']=cl_rep['1.0']['f1-score']
        stats[iter]['FES'][p]['pre_avg']=cl_rep['macro avg']['precision']
        stats[iter]['FES'][p]['rec_avg']=cl_rep['macro avg']['recall']
        stats[iter]['FES'][p]['f1_avg']=cl_rep['macro avg']['f1-score']
        save_stats(stats,args)

###################################
# 8.
# Run the experiment
###################################

for iter in range(args.n_iter):
    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=args.test_size)
    X_test_tensor, y_test_tensor = tensorize(X_test, y_test, args)
    exp_step_1(X_train,y_train,X_test_tensor,y_test_tensor,args,stats,iter)
    exp_step_mp(X_train,y_train,X_test_tensor,y_test_tensor,args,stats,iter)
    exp_step_fes(X_train,y_train,X_test_tensor,y_test_tensor,args,stats,iter)