import warnings
import os
import sys
from data_reduction.statistic import srs_selection, prd_selection
from data_reduction.geometric import clc_selection, mms_selection, des_selection
from data_reduction.ranking import phl_selection, nrmd_selection, psa_selection
from data_reduction.wrapper import fes_selection
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #evitar warnings, info tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import io
import time
import math
from codecarbon import OfflineEmissionsTracker
import shutil
from PIL import Image
import json
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import cKDTree
import argparse
import pandas as pd

metodosPosibles=["NINGUNO","SRS","DES","NRMD","MMS","PSA","RKMEANS","PRD","PHL","FES"]
    
def PathsImagenesCarpeta(ruta):
    paths_imagenes = []
    paths_imagenes_only = []
    ruta_nueva = 'yolov5/wheelchair-detection-1/train/imagesTodas'
    if not os.path.exists(ruta_nueva):
        ruta_carpeta = ruta +'/images'
    else:
        ruta_carpeta = ruta +'/imagesTodas'
    for root, dirs, files in os.walk(ruta_carpeta):
        for file in files:
            if file.endswith('.jpg'):
                path_imagen = ruta_carpeta + "/" + file
                paths_imagenes.append(path_imagen)
                paths_imagenes_only.append(file)
    print(f'Existen {len(paths_imagenes)} imágenes en la ruta {ruta}')
    return paths_imagenes, paths_imagenes_only

def categorize_files(ruta_carpeta):
    categorias = []
    for root, dirs, files in os.walk(ruta_carpeta):
        for file in files:
            if file.endswith('.txt'):
                path_archivo = os.path.join(root, file)
                with open(path_archivo, 'r') as archivo:
                    lineas = archivo.readlines()
                    # Category 0: Only one object: Wheelchair
                    if len(lineas) == 1 and lineas[0].startswith('1'):
                        categorias.append(0)
                    #Category 1: Only one object: Person
                    elif len(lineas) == 1 and lineas[0].startswith('0'):
                        categorias.append(1)
                    # Category 2: Multiple objects combining chairs and people
                    elif len(lineas) > 1 and any(line.startswith('0') for line in lineas) and any(line.startswith('1') for line in lineas):
                        categorias.append(2)
                    # Category 3: Multiple items, all wheelchairs
                    elif len(lineas) > 1 and all(line.startswith('1') for line in lineas):
                        categorias.append(3)
                    # Multiple objects, all people, also category 1
                    elif len(lineas) > 1 and all(line.startswith('0') for line in lineas):
                        categorias.append(1) 
    return categorias

def representative_kmeans(X,categorias,perc):
    n_classes = np.unique(categorias).shape[0]
    kmeans = KMeans(n_clusters=n_classes) 
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    indices = np.arange(0,X.shape[0])
    indicesElegidos=[]
    perc = perc
    for i in range(kmeans.n_clusters):  
        cluster_center = kmeans.cluster_centers_[i]
    
        distances = []
        for j, label in enumerate(cluster_labels):
            if label == i:
                dist = np.linalg.norm(X[j] - cluster_center)
                distances.append((j, dist))
    
        distances.sort(key=lambda x: x[1])
        num_representatives = min(int(int(X.shape[0]*perc)/n_classes), len(distances)) 
        indicesElegidos.extend([indices[idx] for idx, _ in distances[:num_representatives]])
    
    return indicesElegidos
    
def rkmeans(paths_imagenes,tensor_YOLO,y,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio=time.time()
    indices = representative_kmeans(tensor_YOLO,y,perc)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    calcular_epsilon_representatividad(tensor_YOLO,np.array(y),tensor_YOLO[indices],np.array(y)[indices])
    representative_images = [paths_imagenes[indice] for indice in indices]
    
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de RKMEANS con una reduccion a {perc} fue de: {tiempo_transcurrido} segundos")

    paths_imagenes_RepresentativeKMeans=np.array(representative_images)
    print("Hemos pasado de ", tensor_YOLO.shape[0] , " a " , paths_imagenes_RepresentativeKMeans.shape[0])
    return paths_imagenes_RepresentativeKMeans
    
class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.labels)

class MiModelo(nn.Module):
    def __init__(self,l):
        super(MiModelo, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 32, 5, 2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, l)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x,dim=1)
        return x


def train_step(train_loader, model, args, criterion, optimizer):
    model = model.to(args.device)
    model.train() 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad() 
        output = model(data)
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
        
            outputs = model(batch_X.to(args.device))
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == batch_y).tolist()
            for j in range(len(correct)):
                indice = i * batch_size + j
                if indice > n_y:
                    continue
                forgetting_events[indice] += 1 if current_accuracy[indice] > correct[j] else 0
                current_accuracy[indice] = correct[j]
                
def train_fes(X,y,model,criterion,optimizer,args,perc):
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    n_y = len(y)
    current_accuracy = np.zeros(n_y, dtype=np.int32)
    forgetting_events = np.zeros(n_y, dtype=np.int32)
    print("Epochs before reduction:")
    for i in range(args.initial_epochs):
        print(f"\rEpoch {i}", end='', flush=True)
        train_step(train_loader, model, args, criterion, optimizer)
        forgetting_step(model, current_accuracy, forgetting_events, X, y, args)
    X_red, y_red = fes_selection(y,current_accuracy, forgetting_events,perc,args.initial_epochs, X=X)
    # X_red, y_red = fes_classwise(X,y,current_accuracy, forgetting_events,args)
    # train_dataset_red = TensorDataset(X_red, y_red)
    # train_loader_red = DataLoader(train_dataset_red, batch_size=args.batch_size, shuffle=True)
    # print("\n Epochs after reduction:")
    # for i in range(args.initial_epochs,args.total_epochs):
    #     print(f"\rEpoch {i}", end='', flush=True)
    #     train_step(train_loader_red, model, args, criterion, optimizer)
    return X_red, y_red
    
def fes(paths_imagenes,perc,tensor_YOLO,categorias):
    trainImagesPath = 'yolov5/wheelchair-detection-1/train/imagesTodas'
    trainImages = [os.path.join(trainImagesPath,path) for path in os.listdir(trainImagesPath)]
    trainLabelsPath = 'yolov5/wheelchair-detection-1/train/labels'
    trainLabels = categorize_files(trainLabelsPath)
    numCat = np.unique(trainLabels).shape[0]
    
    tensor = torch.zeros((len(trainImages),3,416,416))
    i=0
    for path in trainImages:
      img = preprocess_img_yolo(path)
      tensor[i,:,:,:] = img
      i += 1
    parser = argparse.ArgumentParser(description='Arguments for the experiments')

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
        '--no_dropout', action='store_true', default=False, help='remove dropout')
    parser.add_argument(
        '--dropout_prob',
        type=float,
        default=0.33,
        metavar='M',
        help='Dropout probability (default: 0.33)')
    parser.add_argument(
        '--total_epochs',
        type=int,
        default=15,
        metavar='N',
        help='number of epochs to train (default: 200)')
    parser.add_argument(
        '--initial_epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train before reduction (default: 50)')
    parser.add_argument(
        '--reduction_ratio',
        type=float,
        default=0.5,
        metavar='perc',
        help='reduction percentage (default: 0.5)')
    parser.add_argument(
        '--n_iter',
        type=int,
        default=10,
        metavar='N_iter',
        help='number of iterations of the experiment (default: 10)')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to do the computations. Can be cou or cuda (default: cpu)')
    args = parser.parse_args([
    '--learning_rate','0.01',
    '--momentum','0.5',
    '--batch_size','15',
    '--no_dropout',
    '--dropout_prob', '0.33',
    '--total_epochs','10',
    '--initial_epochs','10',
    '--reduction_ratio',str(perc),
    '--n_iter','15',
    '--device', 'cpu'
    ])
    model = MiModelo(numCat)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio=time.time()
    X_res, y_res= train_fes(tensor,torch.tensor(trainLabels),model,criterion,optimizer,args,perc)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),tensor_YOLO[indices],np.array(categorias)[indices])
    find_epsilon(tensor_YOLO,np.array(categorias),tensor_YOLO[indices],np.array(categorias)[indices])
    paths_imagenes_reducidas = np.array(paths_imagenes)[indices]
    print(f'Hemos pasado de {len(paths_imagenes)} muestras a {len(paths_imagenes_reducidas)} muestras con el método Forgetting Event Score y una reducción a {1 - perc}')
    return paths_imagenes_reducidas
    
def obtenIndicesSeleccionados(full_tensor,reduce_tensor):
    indices = []
    for i, fila in enumerate(full_tensor):
        for fila_res in reduce_tensor:
            if np.array_equal(fila, fila_res):
                indices.append(i)
                break
    return indices

def find_epsilon(X,y,X_res,y_res):
    inicio = time.time()
    epsilon = 0
    classes = np.unique(y)
    for cl in classes:
        A = X_res[y_res==cl]
        if A.shape[0] > 0:
            B = X[y==cl]
            kdtree = cKDTree(A)
            epsilon = max(epsilon,max(kdtree.query(B,p=np.inf)[0]))
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo en calcular la epsilon-representatividad(Metodo Javi con D.Infinito y cKDTree) ({epsilon}) ha sido de {tiempo_transcurrido} segundos")
    return epsilon

def calcular_epsilon_representatividad(X,y,X_res,y_res):
    
    inicio = time.time()
    categorias_X = np.unique(y)  # Obtener las categorías únicas del dataset completo(X,y)
    
    max_distancias = {} # Creamos el diccionario donde vamos a guardar el maximo de las distancias de los puntos no presentes de dataset completo en dataset reducido con los puntos de esa categoria si presentes en el dataset reducido
    
    for categoria in categorias_X:
        # Filtrar puntos de la categoría actual en el dataset completo no presentes en el dataset2
        X1 = X[(y == categoria)] 
        indices = ~np.isin(X1,X_res[(y_res == categoria)]).all(axis=1)
        puntos_categoria_no_seleccionados = X1[indices]
    
        if len(puntos_categoria_no_seleccionados) > 0:
            distancias_categoria = []
    
            # Calcular la distancia euclidiana para cada punto de la categoría no seleccionado en dataset reducido con los si seleccionados.
            for punto in puntos_categoria_no_seleccionados:
                if X_res[y_res == categoria].shape[0] > 0:
                    distancias_punto = euclidean_distances([punto], X_res[y_res == categoria])
                    min_distancia = np.min(distancias_punto)
                    distancias_categoria.append(min_distancia)
    
            # Quedarse con el máximo de las distancias obtenidas para la categoría actual
            if len(distancias_categoria) > 0:
                max_distancias[f'Categoria {categoria}'] = np.max(distancias_categoria)
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo en calcular la epsilon-representatividad (Metodo Victor con D.Euclidea) ({max(max_distancias.values())}) ha sido de {tiempo_transcurrido} segundos")
    # Devolver el máximo de los valores obtenidos para cada categoría
    return max(max_distancias.values()),max_distancias
    
def srs(paths_imagenes,tensor_YOLO,categorias,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = srs_selection(tensor_YOLO,np.array(categorias),perc)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de SRS con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas

def des(paths_imagenes,tensor_YOLO,categorias,perc):
    if perc > 0 and perc > 0.3:
        perc_base = 0.2
    elif perc > 0 and perc < 0.3:
        perc_base = 0.05
    print(f"perc_base = {perc_base}")
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = des_selection(tensor_YOLO,np.array(categorias),perc,perc_base)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de DES con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas

def nrmd(paths_imagenes,tensor_YOLO,categorias,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = nrmd_selection(tensor_YOLO,np.array(categorias),perc,"SVD_python")
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de NRMD con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas


def phl(paths_imagenes,tensor_YOLO,categorias,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = phl_selection(tensor_YOLO,np.array(categorias),4,perc,"multiDim",1,"representative")
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de PHL con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas

def mms(paths_imagenes,tensor_YOLO,categorias,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = mms_selection(tensor_YOLO,np.array(categorias),perc)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de MMS con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas

def psa(paths_imagenes,tensor_YOLO,categorias,perc):
    inicio = time.time()
    X_res, y_res = psa_selection(tensor_YOLO,np.array(categorias),perc,5)
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de PSA con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas

def prd(paths_imagenes,tensor_YOLO,categorias,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = prd_selection(tensor_YOLO,np.array(categorias),perc,3,"osqp")
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de PRD con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas

def preprocess_img_yolo(img_path):
  img = image.load_img(img_path)
  x = torchvision.transforms.ToTensor()(img)
  x = torch.unsqueeze(x,0)
  return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetCarpeta',default="yolov5/wheelchair-detection-1/train", type=str, help='Folder where is located the dataset to reduce')
    parser.add_argument('--name', default='SRS', type=str, help='Reduction method to apply')
    parser.add_argument('--perc', default='0.5', type=float, help="Reduction rate to apply (between 0 and 1)")

    args = parser.parse_args()
    metodo = args.name
    perc = args.perc
    print(os.getcwd())
    ruta_carpeta = "yolov5/wheelchair-detection-1/train/images"  # Reemplaza con la ruta correcta
    ruta_nueva = 'yolov5/wheelchair-detection-1/train/imagesTodas'
    if not os.path.exists(ruta_nueva):
      os.rename(ruta_carpeta, ruta_nueva)
      os.makedirs(ruta_carpeta)
    else:
      shutil.rmtree(ruta_carpeta)
      os.makedirs(ruta_carpeta)
            
    if metodo not in metodosPosibles:
        raise ValueError("The chosen reduction method(--name) is not among the possible ones: ",metodosPosibles)
    elif metodo == "NINGUNO":
        print("You have not selected any method, so you are going to train with the complete training set.")
            
        archivos = os.listdir(ruta_nueva)
        print("Number Original Files", len(archivos))
        for archivo in archivos:
            if archivo.endswith(".jpg"):
                ruta_archivo = os.path.join(ruta_nueva, archivo)
                ruta_archivo_nueva = os.path.join(ruta_carpeta, archivo)
                shutil.copy(ruta_archivo,ruta_archivo_nueva)
        
        print("The training set has a size of: ", len(os.listdir(ruta_nueva)))
        
    else:
        print("Selected Method: ", metodo)

        if perc < 0 or perc > 1:
            raise ValueError("The rate of reduction(--perc) should be between 0 and 1")
        else:
            print("Selected reduction rate: ", perc)
    
            paths_imagenes, paths_imagenes_only = PathsImagenesCarpeta(args.datasetCarpeta)
            categorias = categorize_files(args.datasetCarpeta)
        
            tensor = torch.zeros(len(paths_imagenes),768)
        
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True,verbose=False)
            backbone = model.model.model.model[0:10]
            
            i=0
            for path in tqdm(paths_imagenes):
              img = preprocess_img_yolo(path)
              features = backbone(img.to('cuda'))
              x = torch.nn.AdaptiveAvgPool2d(1)(features)
              x = torch.squeeze(x)
              tensor[i,:] = x
              i+=1
            
            tensor_YOLO = tensor.numpy()
        
            if metodo == "RKMEANS":
                paths_imagenes_seleccionadas = rkmeans(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "SRS":
                paths_imagenes_seleccionadas = srs(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "DES":
                paths_imagenes_seleccionadas = des(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "NRMD":
                paths_imagenes_seleccionadas = nrmd(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "PHL":
                paths_imagenes_seleccionadas = phl(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "PSA":
                paths_imagenes_seleccionadas = psa(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "MMS":
                paths_imagenes_seleccionadas = mms(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "PRD":
                paths_imagenes_seleccionadas = prd(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "FES":
                paths_imagenes_seleccionadas = fes(paths_imagenes_only,perc,tensor_YOLO,categorias)
                
            archivos = os.listdir(ruta_nueva)
            print("Number of original files: ", len(archivos))
            for archivo in archivos:
                if archivo.endswith(".jpg") and archivo in paths_imagenes_seleccionadas:
                    ruta_archivo = os.path.join(ruta_nueva, archivo)
                    ruta_archivo_nueva = os.path.join(ruta_carpeta, archivo)
                    shutil.copy(ruta_archivo,ruta_archivo_nueva)
            
            print("Process completed.")
            print("Number of original files:", len(os.listdir(ruta_nueva)))
            print("Files after using", metodo , " reduction method and a percentage of reduction of ", perc, ": ",  len(os.listdir(ruta_carpeta)))

if __name__ == "__main__":
    main()