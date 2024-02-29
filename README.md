# Experiments

This repository contains the files necessary to run the experiments performed in deliverable 6.2 for the European Project REXASI-PRO (REliable & eXplainable Swarm Intelligence for People with Reduced mObility) (HORIZON-CL4-HUMAN-01 programme under grant agreement nº101070028). It has been created by the CIMAgroup research team at the University of Seville, Spain.

First, we must install the data reduction repository created by the CIMAgroup research team by following the instructions in https://github.com/Cimagroup/SurveyGreenAI.


## Experiments Data Reduction for tabular data



## Experiments Data Reduction for object detection

In addition, we need to install all the necessary python dependencies to run these experiments with the next command, but first you need to enter in the folder ObjectDetection:

```bash
pip install -r requeriments.txt
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
