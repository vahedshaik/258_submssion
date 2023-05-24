from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2 import model_zoo
from detectron2.detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.detectron2.data import MetadataCatalog

import cv2
import numpy as np

class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    def detect(self, imagePath):
        im = cv2.imread(imagePath)
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                       scale=1.2,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        instances = outputs["instances"]
        labels = instances.pred_classes.tolist()  

        for label in labels:
            class_name = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[label]  
            print("Bounding box label:", class_name)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        v.save('../../images/output.jpg')
        return class_name
        # cv2.imshow("Image", v.get_image()[:, :, ::-1])
        # cv2.waitKey(0)

