# Experiments

This repository contains the files necessary to run the experiments performed for the European Project REXASI-PRO (REliable & eXplainable Swarm Intelligence for People with Reduced mObility) (HORIZON-CL4-HUMAN-01 programme under grant agreement nº101070028). It has been created by the CIMAgroup research team at the University of Seville, Spain.

First, we must install the data reduction repository created by the CIMAgroup research team by following the instructions in https://github.com/Cimagroup/SurveyGreenAI. This experiments has been developed in Ubuntu 20.04.


## Experiments Data Reduction for tabular data

In the TabularData folder we have the necessary code to run our experiments on data reduction for tabular data classification. The first experiment is done with the Collision dataset, provided by our colleagues Maurizio Mongelli and Sara Narteni from CNR. The second experiment is done with the Dry Bean dataset, available in UCIML repository. Both experiments have the same structure.

### Tabular Data experiments code

The code to run the experiments can be found in the files "collision/experiment_collision.py" and "drybean/experiment_drybean.py". These two python files are divided into the following sections:

0. Import libraries 

1. Load, scale and shuffle the dataset

In this section, the dataset is loaded, scaled and randomly shuffled.

2. Define the arguments object

In this section an arguments object called *args* is defined. This arguments object contains parameters such as learning rate, momentum, batch size, dropout probability, the size of test set, the device where the computations ara made (cpu, cuda...), the number of epochs to train the model, the number of epochs to calculate the forgetting events (just for FES method), the number of iterations of the experiment, the country where the experiments are being done (this is relevant to estimate the carbon emission) and the name of the file where the results will be saved.

3. Define the neural arquitecture

In this section the arquitecture of the neural network is defined. This implementation uses the torch library, so the neural network is a torch class called *NeuralNetwork*.

4. Create the statistics dictionary

In this section the dictionary where the results are saved is created. It is called *stats* and it has a nested structure. In the first level one can access each iteration of the experiment. On the second level one can access the results for each data reduction method. On the third level one can access the results for a specific reduction ratio. On the fourth level one can access the different statistics for each iteration, method and reduction ratio. These statistics are: reduction + training time, reduction + training carbon emission, $\varepsilon$-representativeness of the reduced dataset, accuracy, precision for each class and macro average, recall for each class and macro average, and F1-score for each class and macro average.  

5. Functions to train the network

In this section some auxiliary functions are defined. The function *create_new_model* builds a new instance of the *NeuralNetwork* class, together with a loss function and an optimizer function. The function *tensorize* transforms a dataset $X,y$ into tensors and sends it to the device specified in *args*. The function *save_stats* saves the dictionary *stats* into the specified file. The function *train_step* performs a training step for our neural network. The function *train_model* creates a DataLoader object and iterates the function *train_step* as many times as specified. The function *forgetting_step* counts the forgetting events after an epoch. The function *train_fes* performs the FES selection and train the neural network all together. The function *predict* calculates the prediction given by the neural network for an example.

6. Function to comprise all reduction methods

In this section a function called *reduce* is defined. It comprises all the model-independent data reduction methods (all but FES) in a single function. 

7. Functions to perform the experiment steps

In this section three functions are defined to divide the experiment into three steps. The first one, *exp_step_1*, trains the model over the whole training set, evaluates the model over the test set, and saves the results. The second one, *exp_step_mp*, reduces the training set for each method (except FES) and reduction ratio, trains the model over the reduced dataset, evaluates the model over the test set, and saves the results. The third function, *exp_step_fes*, trains the model and reduces the dataset with FES, evaluates the model over the test set, and saves the results.

8. Run the experiment

In this section the experiment is run. For each iteration, the dataset is splitted into training and test set, and performs the three functions defined in Section 7, *exp_step_1*, *exp_step_mp* and *exp_step_fes*.   


## Experiments Data Reduction for object detection

In addition, we need to install all the necessary python dependencies to run these experiments with the next command, but first you need to enter in the folder ObjectDetection:

```bash
pip install -r requeriments.txt  #on Ubuntu
```
or, in case you are in Windows:

```bash
pip install -r requerimentsWindows.txt #on Windows
```

### Dataset Roboflow

In order to perform the experiments on the Roboflow dataset, you must use Yolov5DatasetRoboflow.ipynb, where you can choose the reduction method, as well as the reduction percentage.

Pd: If you want to select a reduction rate of 75%, you must enter 0.25 in perc, in the notebook Yolov5DatasetRoboflow.ipynb. If you want to select a specific method of reduction, you can choose one of those listed in Table 1. For example, if you want to apply PRD with a reduction rate of 75 percent, you must run: !python ReductionDatasetRoboflow.py --name "PRD" --perc 0.25

### Dataset Mobility Aid 

In order to perform the experiments on the Mobility Aid dataset, you must use Yolov5DatasetMobilityAid.ipynb, where you can choose the reduction method, as well as the reduction percentage.

To make it work, you must download the following files that you can find at http://mobility-aids.informatik.uni-freiburg.de/ and save them in the DatasetMobilityAid folder.

  - RGB images
  - Annotations RGB
  - Annotations RGB test set 2
  - image set textfiles
  
Once downloaded, you must go to the DataFormatYolov5.ipynb file inside the DatasetMobilityAid folder and run it completely, so that these images are in the proper YoloV5 format, and you can now use Yolov5DatasetMobilityAid.ipynb.

Pd: If you want to select a reduction rate of 75%, you must enter 0.25 in perc, in the notebook Yolov5DatasetMobilityAid.ipynb. If you want to select a specific method of reduction, you can choose one of those listed in Table 1. For example, if you want to apply PRD with a reduction rate of 75 percent, you must run: !python ReductionDatasetMobilityAid.py --name "PRD" --perc 0.25

| Reduction Method    | Name you must enter in method in the  notebook |
|---------------------|--------------------------|
| Full train dataset            | NONE |
| Stratified Random Sampling           | SRS |
| Distance-Entropy Selection           | DES |
| Numerosity Reduction by Matrix Decomposition            | NRMD |
| MaxMin Selection            | MMS |
| Representative KMeans            | RKMEANS |
| PRotoDash           | PRD |
| Persistence Homology Landmarks Selection          | PHL |
| Forgetting Events Score            | FES |

Table 1: Reduction methods list and how you have to call them in the notebook experiments
