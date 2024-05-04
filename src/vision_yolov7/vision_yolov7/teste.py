import rclpy
from rclpy.node import Node
from custom_interfaces.msg import Vision, VisionVector
import sys
sys.path.insert(0, './src/vision_yolov7/vision_yolov7')
import numpy as np
import cv2
from pathlib import Path
import torch
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load

# Carregar informações do arquivo yaml
with open('src/vision_yolov7/vision_yolov7/data.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Obter o número de classes e os nomes das classes
num_classes = config['nc']
class_names = config['names']

PATH_TO_WEIGHTS = 'src/vision_yolov7/vision_yolov7/peso_tiny/best_localization.pt'

class LandmarkDetection(Node):

    def __init__(self):
        super().__init__('landmark_detection')
        self.publisher_landmarks = self.create_publisher(VisionVector, '/landmark_position', 10)
        self.weights = PATH_TO_WEIGHTS
        self.detect_landmarks()

    def detect_landmarks(self):
        set_logging()
        device = select_device('cpu')  # Pode alterar para 'cuda' se tiver GPU
        
        # Carregar modelo
        model = attempt_load(self.weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(640, s=stride)

        dataset = LoadStreams('/dev/camera', img_size=imgsz, stride=stride)

        names = model.names

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img)[0]

            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=True)

            im0, frame = im0s[0].copy(), dataset.count

            det = pred[0]

            if len(det):  # Entra aqui se detectou algum landmark
                det[:, :4] = det[:, :4].clamp(min=0, max=im0s.shape[2])
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=(0, 255, 0), line_thickness=3)

            cv2.imshow('Landmark Detection', im0)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    landmark_detection = LandmarkDetection()
    rclpy.spin(landmark_detection)
    landmark_detection.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
