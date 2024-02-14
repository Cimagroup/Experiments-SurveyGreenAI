import warnings
import os
import sys
from data_reduction.statistic import srs_selection, prd_selection
from data_reduction.geometric import clc_selection, mms_selection, des_selection
from data_reduction.ranking import phl_selection, nrmd_selection, psa_selection
from data_reduction.wrapper import fes_selection
# sys.path.append("Original_repositories/pytorch_influence_functions")
# import pytorch_influence_functions as ptif
# sys.path.append("..")
warnings.filterwarnings("ignore")

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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import io
import time
from codecarbon import OfflineEmissionsTracker
import shutil
from PIL import Image
import json
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import cKDTree
import argparse
import pandas as pd

metodosPosibles=["NINGUNO","SRS","DES","NRMD","MMS","PSA","RKMEANS","PRD","PHL","IF","FES"]

def PathsImagenesCarpeta(ruta):
    paths_imagenes = []
    paths_imagenes_only = []
    ruta_nueva = 'Dataset2/dataYOLOv5/train/imagesTodas'
    if not os.path.exists(ruta_nueva):
        ruta_carpeta = ruta +'/images'
    else:
        ruta_carpeta = ruta +'/imagesTodas'
    for root, dirs, files in os.walk(ruta_carpeta):
        for file in files:
            if file.endswith('.png'):
                path_imagen = ruta_carpeta + "/" + file
                paths_imagenes.append(path_imagen)
                paths_imagenes_only.append(file)
    print(f'Existen {len(paths_imagenes)} imágenes en la ruta {ruta_carpeta}')
    return paths_imagenes, paths_imagenes_only

def categorizar_archivos(ruta):
    categorias = []
    NumeroPersonas=0
    NumeroPersonasEnSillas=0
    NumeroPersonasEmpujandoSilla=0
    NumeroPersonasMuletas=0
    NumeroPersonasAndador=0
    NumeroNegativo=0
    for root, dirs, files in os.walk(ruta+"/labels"):
        for file in tqdm(files):
            if file.endswith('.txt'):
                path_archivo = os.path.join(root, file)
                with open(path_archivo, 'r') as archivo:
                    lineas = archivo.readlines()
                    for line in lineas:
                        if line.startswith('0'):
                            NumeroPersonas += 1
                        elif line.startswith('1'):
                            NumeroPersonasEnSillas += 1
                        elif line.startswith('2'):
                          NumeroPersonasEmpujandoSilla += 1
                        elif line.startswith('3'):
                          NumeroPersonasMuletas += 1
                        elif line.startswith('4'):
                          NumeroPersonasAndador += 1
                        elif line.startswith('-1'):
                          NumeroNegativo += 1
                    # Categoría 1: Solo un objeto, persona sin problema movilidad
                    if len(lineas) == 1 and (lineas[0].startswith('0') or lineas[0].startswith('2')):
                        categorias.append(0)
                     #Categoría 2: Solo un objeto, persona con problema movilidad
                    elif len(lineas) == 1 and (lineas[0].startswith('1') or lineas[0].startswith('3') or lineas[0].startswith('4')):
                        categorias.append(1)
                    # Categoria 3: Combinacion objetos, todos personas sin problema movilidad
                    elif len(lineas) > 1 and all(line.startswith('0') or line.startswith('2') for line in lineas):
                        categorias.append(2)
                    # Categoria 4: Combinacion objetos, todos personas con problema movilidad
                    elif len(lineas) > 1 and all(line.startswith('1') or line.startswith('3') or line.startswith('4') for line in lineas):
                        categorias.append(3)
                    #Categoria 5: Combinación objetos, personas con y sin problemas de movilidad
                    else:
                        categorias.append(4)
    print(f'Hay {NumeroPersonas} de personas, {NumeroPersonasEnSillas} en sillas de ruedas, {NumeroPersonasEmpujandoSilla} empujando sillas, {NumeroPersonasMuletas} con muletas y {NumeroPersonasAndador} con andador')
    return categorias

def categorizar_archivos_influence(ruta_carpeta):
    categorias = []
    for root, dirs, files in os.walk(ruta_carpeta):
        for file in tqdm(files):
            if file.endswith('.txt'):
                path_archivo = os.path.join(root, file)
                with open(path_archivo, 'r') as archivo:
                    lineas = archivo.readlines()
                    # Categoría 1: Solo un objeto, persona sin problema movilidad
                    if len(lineas) == 1 and (lineas[0].startswith('0') or lineas[0].startswith('2')):
                        categorias.append(0)
                     #Categoría 2: Solo un objeto, persona con problema movilidad
                    elif len(lineas) == 1 and (lineas[0].startswith('1') or lineas[0].startswith('3') or lineas[0].startswith('4')):
                        categorias.append(1)
                    # Categoria 3: Combinacion objetos, todos personas sin problema movilidad
                    elif len(lineas) > 1 and all(line.startswith('0') or line.startswith('2') for line in lineas):
                        categorias.append(2)
                    # Categoria 4: Combinacion objetos, todos personas con problema movilidad
                    elif len(lineas) > 1 and all(line.startswith('1') or line.startswith('3') or line.startswith('4') for line in lineas):
                        categorias.append(3)
                    #Categoria 5: Combinación objetos, personas con y sin problemas de movilidad
                    else:
                        categorias.append(4)
    return categorias

def representative_kmeans(X,categorias,perc):
    n_classes = np.unique(categorias).shape[0]
    kmeans = KMeans(n_clusters=n_classes)  # Reemplaza con el número de clusters deseado
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    indices = np.arange(0,X.shape[0])
    indicesElegidos=[]
    perc = perc
    for i in range(kmeans.n_clusters):  # Iterar sobre cada cluster
        cluster_center = kmeans.cluster_centers_[i]  # Obtener el centroide del cluster i
    
        # Calcular la distancia euclidiana de cada imagen al centroide del cluster
        distances = []
        for j, label in enumerate(cluster_labels):
            if label == i:
                dist = np.linalg.norm(X[j] - cluster_center)
                distances.append((j, dist))
    
        # Ordenar las imágenes del cluster por su distancia al centroide y seleccionar las más cercanas
        distances.sort(key=lambda x: x[1])
        num_representatives = min(int(int(X.shape[0]*perc)/n_classes), len(distances)) 
        indicesElegidos.extend([indices[idx] for idx, _ in distances[:num_representatives]])
        
    return indicesElegidos
    
def rkmeans(paths_imagenes,tensor_YOLO,y,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio=time.time()
    indices = representative_kmeans(tensor_YOLO,y,perc)
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
    calcular_epsilon_representatividad(tensor_YOLO,np.array(y),tensor_YOLO[indices],np.array(y)[indices])
    find_epsilon(tensor_YOLO,np.array(y),tensor_YOLO[indices],np.array(y)[indices])
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
        self.fc1 = nn.Linear(64 * 59 * 32, 128)
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

# def influence(paths_imagenes,perc,tensor_YOLO,categorias):
#     trainImagesPath = 'Dataset2/dataYOLOv5/train/imagesTodas'
#     testImagesPath = 'Dataset2/dataYOLOv5/test/images'
#     trainImages = [os.path.join(trainImagesPath,path) for path in os.listdir(trainImagesPath)]
#     testImages = [os.path.join(testImagesPath,path) for path in os.listdir(testImagesPath)]
#     trainLabelsPath = 'Dataset2/dataYOLOv5/train/labels'
#     trainLabels = categorizar_archivos_influence(trainLabelsPath)
#     testLabelsPath = 'Dataset2/dataYOLOv5/test/labels'
#     testLabels= categorizar_archivos_influence(testLabelsPath)
#     transform = transforms.Compose([transforms.ToTensor(),])
#     train_dataset = ImageDataset(trainImages, trainLabels, transform)
#     test_dataset = ImageDataset(testImages,testLabels,transform)
#     trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#     numCat = np.unique(trainLabels).shape[0]
#     ptif.init_logging()
#     config = {
#             'outdir': 'outdir_wheelchair',
#             'seed': 42,
#             'gpu': -1,
#             'num_classes': numCat,
#             'test_sample_num': 1,
#             'test_start_index': 0,
#             'recursion_depth': 1,
#             'r_averaging': 1, #recursion_depth * r_averaging should be as the size of train data
#             'scale': None,
#             'damp': None,
#             'calc_method': 'img_wise',
#             'log_filename': None,
#         }
#     model=MiModelo(numCat)
#     maximo = config["num_classes"]*config["test_sample_num"]
#     tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
#     tracker.start()
#     inicio=time.time()
#     influences = ptif.calc_img_wise(config, model, trainloader, testloader)
#     fin = time.time()
#     tiempo_transcurrido = fin - inicio
#     print(f"El tiempo de ejecución fue de: {tiempo_transcurrido} segundos")
#     with open(f'outdir_wheelchair/influence_results_tmp_0_{config["test_sample_num"]}_last-i_{maximo-1}.json','r') as archivo:
#         influenceList = json.load(archivo)
#         lista = influenceList[f'{maximo-1}']['influence']
#         indicesIF = sorted(range(len(lista)), key=lambda k: lista[k])

#     perc=perc
#     indices = indicesIF[:int(perc*len(paths_imagenes))]
#     print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
#     calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),tensor_YOLO[indices],np.array(categorias)[indices])
#     find_epsilon(tensor_YOLO,np.array(categorias),tensor_YOLO[indices],np.array(categorias)[indices])
#     paths_imagenes_IF=np.array(paths_imagenes)[indices]
#     shutil.rmtree(config["outdir"])
#     print(f'Hemos pasado de {len(paths_imagenes)} muestras a {len(paths_imagenes_IF)} muestras con el método INFLUENCE y una reducción a {perc}')
#     return paths_imagenes_IF

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
                
def train_fes(X,y,model,criterion,optimizer,args):
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
    X_red, y_red = fes_selection(X,y,current_accuracy, forgetting_events,args)
    # X_red, y_red = fes_classwise(X,y,current_accuracy, forgetting_events,args)
    # train_dataset_red = TensorDataset(X_red, y_red)
    # train_loader_red = DataLoader(train_dataset_red, batch_size=args.batch_size, shuffle=True)
    # print("\n Epochs after reduction:")
    # for i in range(args.initial_epochs,args.total_epochs):
    #     print(f"\rEpoch {i}", end='', flush=True)
    #     train_step(train_loader_red, model, args, criterion, optimizer)
    return X_red, y_red        
    
def fes(paths_imagenes,perc,tensor_YOLO,categorias):
    trainImagesPath = 'Dataset2/dataYOLOv5/train/imagesTodas'
    trainImages = [os.path.join(trainImagesPath,path) for path in os.listdir(trainImagesPath)]
    print(f"Realizando método FES... en {len(trainImages)} imagenes")
    trainLabelsPath = 'Dataset2/dataYOLOv5/train/labels'
    trainLabels = categorizar_archivos_influence(trainLabelsPath)
    numCat = np.unique(trainLabels).shape[0]
    
    tensor = torch.zeros((len(trainImages),3,540,960))
    i=0
    for path in tqdm(trainImages):
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
    '--device', 'cuda'
    ])
    model = MiModelo(numCat)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio=time.time()
    X_res, y_res= train_fes(tensor,torch.tensor(trainLabels),model,criterion,optimizer,args)
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
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
    print(f"El tiempo en calcular la epsilon-representatividad ({max(max_distancias.values())}) ha sido de {tiempo_transcurrido} segundos")
    # Devolver el máximo de los valores obtenidos para cada categoría
    return max(max_distancias.values()),max_distancias
    
    
def srs(paths_imagenes,tensor_YOLO,categorias,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = srs_selection(tensor_YOLO,np.array(categorias),perc)
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
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
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = des_selection(tensor_YOLO,np.array(categorias),perc,perc_base)
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de DES con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas

# def dom(paths_imagenes,tensor_YOLO,categorias,perc):
#     inicio = time.time()
#     for epsi in np.arange(0.1,10,0.025):
#        X_res, y_res= dom_selection(tensor_YOLO,np.array(categorias),epsilon=epsi)
#        if X_res.shape[0] / tensor_YOLO.shape[0] > perc-0.01 and X_res.shape[0] / tensor_YOLO.shape[0] < perc+0.01:
#          print(epsi)
#          fin = time.time()
#          tiempo_transcurrido = fin - inicio
#          print(f"El tiempo de ejecución de DOM con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
#          break
    
#     indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
#     calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
#     find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
#     paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
#     return paths_imagenes_reducidas

def nrmd(paths_imagenes,tensor_YOLO,categorias,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = nrmd_selection(tensor_YOLO,np.array(categorias),perc,"SVD_python")
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
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
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
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
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
    fin = time.time()
    tiempo_transcurrido = fin - inicio
    print(f"El tiempo de ejecución de MMS con una reducción a {perc} fue de: {tiempo_transcurrido} segundos")
    indices = obtenIndicesSeleccionados(tensor_YOLO,X_res)
    calcular_epsilon_representatividad(tensor_YOLO,np.array(categorias),X_res,y_res)
    find_epsilon(tensor_YOLO,np.array(categorias),X_res,y_res)
    paths_imagenes_reducidas=np.array(paths_imagenes)[indices]
    return paths_imagenes_reducidas

def psa(paths_imagenes,tensor_YOLO,categorias,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    inicio = time.time()
    X_res, y_res = psa_selection(tensor_YOLO,np.array(categorias),perc,20)
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
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
    print("Emisiones estimadas: ", tracker.stop()*1000, " de gramos de CO2e")
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
    parser = argparse.ArgumentParser(description='Script que realiza operaciones con argumentos.')
    parser.add_argument('--datasetCarpeta',default="Dataset2/dataYOLOv5/train", type=str, help='Carpeta donde se encuentra el dataset a reducir(Entrenamiento')
    parser.add_argument('--name', default='SRS', type=str, help='Metodo de reducción a aplicar')
    parser.add_argument('--perc', default='0.5', type=float, help="Tasa de reducción a aplicar(entre 0 y 1")

    args = parser.parse_args()
    metodo = args.name
    perc = args.perc
    ruta_carpeta = "Dataset2/dataYOLOv5/train/images"  # Reemplaza con la ruta correcta
    ruta_nueva = 'Dataset2/dataYOLOv5/train/imagesTodas'
    
    if metodo not in metodosPosibles:
        raise ValueError("El método de reducción elegido(--name) no está entre uno de los posibles: ",metodosPosibles)
    elif metodo == "NINGUNO":
        print("No has seleccionado ningún metodo, por lo que vas a entrenar con el conjunto de entrenamiento al completo")
        if not os.path.exists(ruta_nueva):
          os.rename(ruta_carpeta, ruta_nueva)
          os.makedirs(ruta_carpeta)
        else:
          shutil.rmtree(ruta_carpeta)
          os.makedirs(ruta_carpeta)
            
        # Obtener la lista de archivos en la carpeta
        archivos = os.listdir(ruta_nueva)
        print("Archivos originales: ", len(archivos))
        # Recorrer los archivos y borrar los que no estén en nombres_viables
        for archivo in archivos:
            if archivo.endswith(".png"):
                ruta_archivo = os.path.join(ruta_nueva, archivo)
                ruta_archivo_nueva = os.path.join(ruta_carpeta, archivo)
                shutil.copy(ruta_archivo,ruta_archivo_nueva)
        
        print("El conjunto de entrenamiento tiene un tamaño de: ", len(os.listdir(ruta_nueva)))
    else:
        print("Metodo Seleccionado: ", metodo)

        if perc < 0 or perc > 1:
            raise ValueError("La tasa de reducción(--perc) debe estar entre 0 y 1")
        else:
            print("Tasa de reducción seleccionada: ", perc)

            paths_imagenes, paths_imagenes_only = PathsImagenesCarpeta(args.datasetCarpeta)
            categorias = categorizar_archivos(args.datasetCarpeta)
        
            tensor = torch.zeros(len(paths_imagenes_only),768)
        
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True,verbose=False)
            backbone = model.model.model.model[0:10]
            
            i=0
            for path in tqdm(paths_imagenes):
              img = preprocess_img_yolo(path)
              features = backbone(img.to('cuda')) # si ponemos img.to('cuda') mucho mas rapido pero hay que tener GPU conectada
              x = torch.nn.AdaptiveAvgPool2d(1)(features)
              x = torch.squeeze(x) # elimina las dimensiones innecesarias, es decir las que tienen 1
              tensor[i,:] = x
              i+=1
            
            tensor_YOLO = tensor.numpy()
            # print(tensor_YOLO.shape)
        
            if metodo == "RKMEANS":
                paths_imagenes_seleccionadas = rkmeans(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "SRS":
                paths_imagenes_seleccionadas = srs(paths_imagenes_only,tensor_YOLO,categorias,perc)
            elif metodo == "DES":
                paths_imagenes_seleccionadas = des(paths_imagenes_only,tensor_YOLO,categorias,perc)
            # elif metodo == "DOM":
            #     paths_imagenes_seleccionadas = dom(paths_imagenes_only,tensor_YOLO,categorias,perc)
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
            # elif metodo == "IF":
            #     paths_imagenes_seleccionadas = influence(paths_imagenes_only,perc,tensor_YOLO,categorias)
            elif metodo == "FES":
                paths_imagenes_seleccionadas = fes(paths_imagenes_only,perc,tensor_YOLO,categorias)
                
            if not os.path.exists(ruta_nueva):
              os.rename(ruta_carpeta, ruta_nueva)
              os.makedirs(ruta_carpeta)
            else:
              shutil.rmtree(ruta_carpeta)
              os.makedirs(ruta_carpeta)
                
            # Obtener la lista de archivos en la carpeta
            archivos = os.listdir(ruta_nueva)
            print("Archivos originales: ", len(archivos))
            # Recorrer los archivos y borrar los que no estén en nombres_viables
            for archivo in archivos:
                if archivo.endswith(".png") and archivo in paths_imagenes_seleccionadas:
                    ruta_archivo = os.path.join(ruta_nueva, archivo)
                    ruta_archivo_nueva = os.path.join(ruta_carpeta, archivo)
                    shutil.copy(ruta_archivo,ruta_archivo_nueva)
            
            print("Proceso completado.")
            print("Archivos originales: ", len(os.listdir(ruta_nueva)))
            print("Archivos tras métodos de reduccion ", metodo , " y porcentage de reducción a un ", perc, ": ",  len(os.listdir(ruta_carpeta)))

if __name__ == "__main__":
    main()