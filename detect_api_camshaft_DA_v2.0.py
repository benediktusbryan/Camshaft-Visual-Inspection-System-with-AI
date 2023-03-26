# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long
# pylint: disable=C0321
from fastapi import FastAPI #
import uvicorn  #
from fastapi import FastAPI, File, UploadFile   #
from typing import List, Optional   #

import argparse #
import shutil   #
from pathlib import Path    #
from pydantic import BaseModel  #

import detectron2
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
import glob
import os
import random
import cv2
import numpy as np
import argparse
import time
from datetime import datetime
import logging

#api
app = FastAPI()
app.logger = logging.getLogger('uvicorn')
# from detectron2.utils.logger import setup_logger
# setup_logger()

annotations_color = [
  (0,0,0),        #black
  (255,0,0),      #red
  (0,255,0),      #green
  (0,0,255),      #blue
  (255,255,0),    #yellow
  (255,0,255),    #purple
  (0,255,255),    #cyan
  (255,255,204),  #ivory
  (255,204,153),  #peach
  (255,153,0),    #orange
  (51,153,102),   #tosca
  (153,204,255),  #sky blue
  (255,153,204),  #pink
  (153,153,255),  #magenta
  (204,255,204),  #light green
  (128,128,0),    #dark yellow
]
predictors = list()
modelNames = list()
modelListDir = "models"
queueDir ='D:/Pre-processing Queue.txt'

#inisialisasi detectron
def initialize(listDir):
    cfg = get_cfg()
    # register_coco_instances("classes1", {}, "classes/kelas.json", "classes")
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TEST = ("classes1",)
    cfg.MODEL.DEVICE = 'cuda'

    modelNames, modelPaths, confThresholds, iouThresholds = getModelList(listDir)
    for i in range(len(modelNames)):
        #Setup per model
        t7 = time.perf_counter()

        modelName = modelNames[i]
        modelPath = modelPaths[i]
        confThreshold = confThresholds[i]
        iouThreshold = iouThresholds[i]

        numberClass, CLASS_NAMES, thing_colors = getModelClasses(modelPath)

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = numberClass + 1
        cfg.MODEL.WEIGHTS = os.path.join(modelPath, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confThreshold    # set the testing threshold for this model
        cfg.IOU_THRESHOLD = iouThreshold

        predictors.append(DefaultPredictor(cfg))
        MetadataCatalog.get(modelName)
        MetadataCatalog.remove(modelName)

        MetadataCatalog.get(modelName).set(thing_classes=CLASS_NAMES).set(thing_colors=thing_colors)
        print(MetadataCatalog.get(modelName))

        t8 = time.perf_counter()
        print('time load model. (%.3fs)' % (t8 - t7))

def getModelList(dirname):
    modelPaths = list()
    confThresholds = list()
    iouThresholds = list()
    try:
        print(dirname + '/model_list.txt')
        model_list_file = open(dirname + '/model_list.txt', "r")
        # numberClass = 0
        for line in model_list_file:
            modelName = line.strip().split(",")[0]
            modelPath = line.strip().split(",")[1]
            confThreshold = float(line.strip().split(",")[2])
            iouThreshold = float(line.strip().split(",")[3])

            modelNames.append(modelName)
            modelPaths.append(modelPath)
            confThresholds.append(confThreshold)
            iouThresholds.append(iouThreshold)

            print('Model Name: {}'.format(modelName))
            print('Model Path: {}'.format(modelPath))
            print('Confidence Threshold: {}'.format(confThreshold))
            print('IOU Threshold: {}'.format(iouThreshold))
        model_list_file.close
    except IOError:
        print("Error: Model List File does not appear to exist or can't be opened.")

    return modelNames, modelPaths, confThresholds, iouThresholds

def getModelClasses(dirname):
    thing_colors = [(0,0,0),]
    CLASS_NAMES = ["defects",]

    try:
        print(dirname + '/classes.txt')
        classes_file = open(dirname + '/classes.txt', "r")
        numberClass = 0
        for line in classes_file:
            if line != "\n" and line != " \n":
                numberClass += 1

            if line.strip() not in CLASS_NAMES:
                CLASS_NAMES.append(line.strip())
                thing_colors.append(annotations_color[numberClass])
        classes_file.close
        print('Total classes: {}'.format(numberClass))
        print('Classes Name: {}'.format(CLASS_NAMES))
        print('Classes Color: {}'.format(thing_colors))
    except IOError:
        print("Error: Class File does not appear to exist or can't be opened.")

    return numberClass, CLASS_NAMES, thing_colors

##detectron detect
def detectronDetect(source, modelName): #
    t1 = time.perf_counter()

    _defectTypesData = list()   #
    _defectDetectedPositionData = list()    #
    #im = cv2.imread(args["img"])
    im = cv2.imread(source)
    
    imageName = "D:/image_test.jpg" #
    cv2.imwrite(imageName, im)
    
    #im = cv2.resize(im, (480,640))
    t3 = time.perf_counter()    #

    test_metadata = MetadataCatalog.get(modelName)
    w = test_metadata.thing_classes

    predictor = predictors[modelNames.index(modelName)]
    outputs = predictor(im)

    t4 = time.perf_counter()
    print('time predictor. (%.3fs)' % (t4 - t3))
    q = outputs["instances"]
    # get prediksi box
    bobox = q.get("pred_boxes")
    # bobox = qq.tensor[0]
    totalDefect = len(q)
    print(f"Total Defect Detected: {str(totalDefect)}")
    ##for data in bobox.tensor.numpy():
    ##    app.logger.info(lokasibox(data))

    for i in range(0, len(q.get("scores"))):
        # app.logger.info("Confindence : ", str(q.get("scores")[i].numpy().item()), "\tClass: ", str(q.get("pred_classes")[i].numpy().item()), "[", w[q.get("pred_classes")[i].numpy().item()], "]")
        conf = q.get("scores")[i].cpu().numpy().item()   #
        nameClass = w[q.get("pred_classes")[i].cpu().numpy().item()]  #

        print(f"Confidence : {str(conf)}\tClass: {nameClass}")   #

        data_xywh = bobox.tensor.cpu().numpy()[i,]    #
        print(lokasibox(data_xywh)) #
        _defectPosition = positionStruct(height=data_xywh[3], width=data_xywh[2], x=data_xywh[0], y=data_xywh[1])
        _defectProbable = probableTypesStruct(confidence=conf, type=nameClass)
        _defectData = detectionsStruct(position=_defectPosition, probableTypes=[_defectProbable])
        _defectDetectedPositionData.append(_defectData)

        _defectType = defectsStruct(defectTypes=nameClass, total=totalDefect) #
        _defectTypesData.append(_defectType)    #

    t5 = time.perf_counter()
    print('time get defect. (%.3fs)' % (t5 - t4))

    v = Visualizer(im[:, :, ::-1],
                        metadata=test_metadata,
                        scale=1,
                        instance_mode=ColorMode.SEGMENTATION
                        )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    t6 = time.perf_counter()
    print('time drawing. (%.3fs)' % (t6 - t5))
    ##cv2.imshow("trial", out.get_image()[:, :, ::-1])
    ##imageName = os.path.join("out/", 'pic-{}.jpg'.format(datetime.now().strftime("%Y%m%d-%I-%M-%S-%p")))

    directoryName = os.path.dirname(source)   #

    imageName = str(Path(directoryName) / Path(source).stem) + "_out.jpg" #
    cv2.imwrite(imageName, out.get_image()[:, :, ::-1])
    t2 = time.perf_counter()
    print(f"output image: {str(Path(imageName).name)}")
    print('Scoring Image %s Done. (%.3fs)' % (str(Path(source).name), t2 - t1))
    ##cv2.waitKey(0)

    # for imageName in glob.glob('/test/*jpg'):
    #     im = cv2.imread(imageName)  # import image satuan disini
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=2)
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imshow(out.get_image()[:, :, ::-1])
    #     cnt = cnt + 1
    #     # cv2.imwrite('' +str(cnt)+'.jpg', out.get_image()[:, :, ::-1])

    responseData = ResponseStruct(defects=_defectTypesData, detections=_defectDetectedPositionData, id=str(Path(source).name), status="Success", timestamp=str(datetime.now().timestamp()))    #

    return responseData

def lokasibox(data):
    return {"x1": data[0], "y1": data[1], "w": data[2] - data[0], "h": data[3] - data[1]}


def DA_PreProcessing(sourceDir):
    t1 = time.perf_counter()
    while 1:
        status, file1 = fileOpen(queueDir, 'a')
        print(status)
        if status == 1:
            break
    processFilename = sourceDir.replace("/","\\\\")
    file1.writelines(processFilename)
    file1.writelines("\n")
    file1.close()

    sourceDirPreProcessing = sourceDir.replace(".jpg","_process.jpg") #
    print("Waiting PreProcessing...")

    t3 = time.perf_counter()
    while 1:
        if os.path.isfile(sourceDirPreProcessing) == 1:
            timeoutProc = False
            break
        t4 = time.perf_counter()
        if (t4 - t3) >= 5.0:
            timeoutProc = True
            print("Timeout PreProcessing")
            break

    t2 = time.perf_counter()
    print('time pre-processing. (%.3fs)' % (t2 - t1))

    return sourceDirPreProcessing, timeoutProc

def fileOpen(fn, mode):
    file = 0
    try:
      file= open(fn, mode)
      return 1, file
    except IOError:
      print("Error: File does not appear to exist or can't be opened.")
      return 0, file

class inspectionParam(BaseModel):
    sourceDir:str
    detectronModel:str
    confidenceThres:float
    IOUThres:float

@app.post('/inspect')
async def inspect(inspection:inspectionParam):
    print(f"Processing image: {inspection.sourceDir}")
    sourceDirPreProcessing, timeoutProc = DA_PreProcessing(inspection.sourceDir)
    if timeoutProc == False:
        i = 0
        while 1:
            try:
                i+=1
                # print (i)
                result = detectronDetect(sourceDirPreProcessing, inspection.detectronModel)
                return result
            except Exception as e:
                print("Error Scoring Detectron")
                print(e)
                if i >= 10:
                    result = ResponseStruct(defects=list(), detections=list(), id=str(Path(sourceDirPreProcessing).name), status=str(e), timestamp=str(datetime.now().timestamp()))
                    return result        
    else:
        result = ResponseStruct(defects=list(), detections=list(), id=str(Path(sourceDirPreProcessing).name), status="Failed. Processing Timeout", timestamp=str(datetime.now().timestamp()))
        return result

#response api
class defectsStruct(BaseModel):
    defectTypes: str
    total: int
class positionStruct(BaseModel):
    height: Optional[float] = 0
    width: Optional[float] = 0
    x: Optional[float] = 0
    y: Optional[float] = 0
class probableTypesStruct(BaseModel):
    confidence: Optional[float] = 0
    type: Optional[str] = ""

class detectionsStruct(BaseModel):
    position: positionStruct
    probableTypes: List[probableTypesStruct]
    properties: Optional[list] = []

class ResponseStruct(BaseModel):
    defects: List[defectsStruct]
    detections: List[detectionsStruct]
    id: str
    status: str
    timestamp: Optional[datetime] = 0

#argument detectron
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument("-mj", "--model_journal", required=False, default="model_final_journal.pth",help="path to trained defects model for journal")
    # ap.add_argument("-ms", "--model_sprocket", required=False, default="model_final_sprocket.pth",help="path to trained hole model for sprocket")
    # #ap.add_argument("-i", "--img", required=True,help="path to input image")
    ap.add_argument("-i", "--img", required=False,default="1.jpg", help="path to input image")   #
    args = vars(ap.parse_args())
    print(args)
    initialize(modelListDir)
    uvicorn.run(app=app, host='0.0.0.0', port=3000)


# from typing import Optional, Union
# def inference(path: str)->Union[defectsStruct, detectionsStruct, str]:
#     """[summary]

#     Args:
#         path (str): [description]

#     Returns:
#         Union[defectsStruct, detectionsStruct, str]: [description]
#     """
#     return defectsStruct()

# test = inference('test')