# Experiments Data Reduction on Images

This repository contains the files necessary to run the experiments performed in deliverable 6.2 for the European Project REXASI-PRO (REliable & eXplainable Swarm Intelligence for People with Reduced mObility) (HORIZON-CL4-HUMAN-01 programme under grant agreement nº101070028). It has been created by the CIMAgroup research team at the University of Seville, Spain.

First, we must install the data reduction repository created by the CIMAgroup research team by following the instructions in https://github.com/Cimagroup/SurveyGreenAI.

In addition we must install all the necessary python dependencies with :

```bash
pip install requeriments.txt
```

## Dataset Roboflow

In order to perform the experiments on the Roboflow dataset, you must use Yolov5Dataset1.ipynb, where you can choose the reduction method, as well as the reduction percentage.

## Dataset Mobility Aid 

In order to perform the experiments on the Mobility Aid dataset, you must use Yolov5Dataset2.ipynb, where you can choose the reduction method, as well as the reduction percentage.

To make it work, you must download the following files that you can find at http://mobility-aids.informatik.uni-freiburg.de/ and save them in the Dataset2 folder.

  - RGB images
  - Annotations RGB
  - Annotations RGB test set 2
  - image set textfiles
  
Once downloaded, you must go to the DataFormatoYolov5.ipynb file inside the Dataset2 folder and run it completely, so that these images are in the proper YoloV5 format, and you can now use Yolov5Dataset2.ipynb.