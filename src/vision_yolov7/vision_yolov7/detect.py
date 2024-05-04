import rclpy
from rclpy.node import Node
from custom_interfaces.msg import Vision, VisionVector
import sys
sys.path.insert(0, './src/vision_yolov7/vision_yolov7')
import numpy as np
import cv2
import torch
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from models.experimental import attempt_load
import yaml

class LandmarkDetection(Node):

    def __init__(self):
        super().__init__('landmark_detection')
        self.publisher_landmarks = self.create_publisher(VisionVector, '/landmark_position', 10)
        self.weights = 'src/vision_yolov7/vision_yolov7/peso_tiny/best_localization.pt'
        self.detect_landmarks()

    def detect_landmarks(self):
        set_logging()
        device = select_device('cpu')

        # Carregar modelo com o peso dos landmarks
        model = attempt_load(self.weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(640, s=stride)

        # Dataset de streaming
        dataset = LoadStreams('/dev/camera', img_size=imgsz, stride=stride)

        names = model.names

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # Executar uma vez

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp32
            img /= 255.0  # 0 - 255 para 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img)[0]

            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=True)

            im0, frame = im0s[0].copy(), dataset.count

            if pred[0] is not None:
                for *xyxy, conf, cls in reversed(pred[0]):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    if names[int(cls)] == "center":
                        # Lógica para processar a detecção do landmark "center"
                        # Aqui você pode adicionar o código para publicar a posição do landmark "center"
                        pass
                    elif names[int(cls)] == "penalti":
                        # Lógica para processar a detecção do landmark "penalti"
                        # Aqui você pode adicionar o código para publicar a posição do landmark "penalti"
                        pass
                    elif names[int(cls)] == "goalpost":
                        # Lógica para processar a detecção do landmark "goalpost"
                        # Aqui você pode adicionar o código para publicar a posição do landmark "goalpost"
                        pass

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
