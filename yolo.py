from sahi import AutoDetectionModel
from sahi.prediction import PredictionResult
from sahi.predict import get_prediction, get_sliced_prediction
from norfair import Detection, Tracker, Video, draw_boxes, draw_tracked_boxes
from torch import cuda
from typing import List
import numpy as np

class YOLO:

    def __init__(self, version='yolov5', weights='best.pt', use_gpu=0):
        """
        Description:
            initializes a new YOLO model
        Parameters:
            version (String)    : the model to be used as the predictor (e.g. 'yolov5', 'yolov8')
            weights (String)    : path to the .pt file containing the weights of the model
            use_gpu (int)       : 1 if a cuda-supported GPU is available, 0 otherwise
        Returns:
            None
        """
        device = 'cpu'
        if use_gpu == 1 and cuda.is_available():
            device = 'cuda:0'
        self.version=version
        self.model = AutoDetectionModel.from_pretrained(
            model_type=version,
            model_path=weights,
            confidence_threshold=0.2,
            device=device)
            
            
    def predict_frame(self, frame, slice=False):
        """
        Description:
            runs the input frame through an inference pass on the class's model
        Parameters:
            frame (String)  : video frame to be processed and tracked
            slice (bool)    : whether the frame should be sliced via sahi for
                              processing (note that this increases accuracy for
                              detecting small objects, but increases inference time)
        Returns:
            List[Detections]: list of norfair Detection objects for the results of
                              running the model on the input frame
        """
        if slice:
            results = get_sliced_prediction(
                frame,
                self.model,
                slice_height=int(frame.shape[1] / 2),
                slice_width=int(frame.shape[1] / 2),
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
        else:
            results = get_prediction(frame, self.model)

        return self._get_detections(results.object_prediction_list)
        
        
    #adapted from https://github.com/tryolabs/norfair/blob/master/demos/sahi/src/demo.py
    def _get_detections(self, object_prediction_list: PredictionResult) -> List[Detection]:
        detections = []
        for prediction in object_prediction_list:
            bbox = prediction.bbox

            detection_as_xyxy = bbox.to_voc_bbox()
            bbox = np.array(
                [
                    [detection_as_xyxy[0], detection_as_xyxy[1]],
                    [detection_as_xyxy[2], detection_as_xyxy[3]],
                ]
            )
            detections.append(
                Detection(
                    points=bbox,
                    scores=np.array([prediction.score.value for _ in bbox]),
                    label=prediction.category.id,
                )
            )
        return detections
         
