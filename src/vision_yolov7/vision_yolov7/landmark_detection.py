import rclpy
from rclpy.node import Node
import math
from custom_interfaces.msg import Vision, VisionVector, VisionVector1, VisionVector2
import sys
sys.path.insert(0, './src/vision_yolov7/vision_yolov7')
import numpy as np
from numpy import random
import cv2
import torch
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from models.experimental import attempt_load
from .ClassConfig import *

THRESHOLD = 0.45

class LandmarkDetection(Node):

    def __init__(self, config):
        super().__init__('landmark_detection')
        self.config = config
        self.publisher_centerlandmark = self.create_publisher(VisionVector, '/centerlandmark_position', 10)
        self.publisher_penaltilandmark = self.create_publisher(VisionVector1, '/penaltilandmark_position', 10)
        self.publisher_goalpostlandmark = self.create_publisher(VisionVector2, '/goalpostlandmark_position', 10)
        self.weights = 'src/vision_yolov7/vision_yolov7/peso_tiny/best_localization.pt'
        self.detect_landmarks()
        self.config = config
        
    def detect_landmarks(self):
        set_logging()
        device = select_device('cpu')
        msg_centerlandmark=VisionVector()
        msg_penaltilandmark=VisionVector1()
        msg_goalpostlandmark=VisionVector2()
        #Load modelo com o peso dos landmarks
        model = attempt_load(self.weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(640, s=stride)
        #Nomes das classes
        names = model.names
        #Cores
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        #Camera
        dataset = LoadStreams('/dev/video0', img_size=imgsz, stride=stride)

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  #Executar uma vez

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  #uint8 to fp32
            img /= 255.0  #0 - 255 para 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img)[0]

            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=True)

            im0, frame = im0s[0].copy(), dataset.count

            msg_centerlandmark.detected = msg_penaltilandmark.detected = msg_goalpostlandmark.detected = False
            msg_centerlandmark.left = msg_penaltilandmark.left = msg_goalpostlandmark.left = False
            msg_centerlandmark.center_left = msg_penaltilandmark.center_left = msg_goalpostlandmark.center_left = False
            msg_centerlandmark.center_right = msg_penaltilandmark.center_right = msg_goalpostlandmark.center_right = False
            msg_centerlandmark.right = msg_penaltilandmark.right = msg_goalpostlandmark.right = False
            msg_centerlandmark.med = msg_penaltilandmark.med = msg_goalpostlandmark.med = False
            msg_centerlandmark.far = msg_penaltilandmark.far = msg_goalpostlandmark.far = False
            msg_centerlandmark.close = msg_penaltilandmark.close = msg_goalpostlandmark.close = False
         
            if pred[0] is not None:
                for *xyxy, conf, cls in reversed(pred[0]):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    if conf>THRESHOLD: #Se confiabilidade maior que 0.45, então detecção considerada válida
                        c1_center = (xyxy[0] + xyxy[2]) / 2
                        c2_center = (xyxy[1] +  xyxy[3]) / 2
                        if names[int(cls)] == "center":
                            #Lógica para processar a detecção do center e publicar posição
                            msg_centerlandmark.detected = True
                            print("Centro do campo detectado '%s'" % msg_centerlandmark.detected)
                            #Centro do campo à esquerda da visão
                            if (int(c1_center) <= self.config.x_left):
                                msg_centerlandmark.left = True
                                self.publisher_centerlandmark.publish(msg_centerlandmark)
                                print("Centro do Campo à Esquerda")
                            #Centro do campo no centro-esquerda da visão
                            elif (int(c1_center) > self.config.x_left and int(c1_center) < self.config.x_center):
                                msg_centerlandmark.center_left = True
                                self.publisher_centerlandmark.publish(msg_centerlandmark)
                                print("Centro do Campo Centralizado à Esquerda")
                            #Centro do campo no centro-direita da visão
                            elif (int(c1_center) < self.config.x_right and int(c1_center) > self.config.x_center):
                                msg_centerlandmark.center_right = True
                                self.publisher_centerlandmark.publish(msg_centerlandmark)
                                print("Centro do Campo Centralizado à a Direita")
                            #Centro do campo à direita da visão
                            else:
                                msg_centerlandmark.right = True
                                self.publisher_centerlandmark.publish(msg_centerlandmark)
                                print("Centro do Campo à Direita")
                            #Centro do campo perto
                            if (int(c2_center) > self.config.y_chute):
                                msg_centerlandmark.close = True
                                self.publisher_centerlandmark.publish(msg_centerlandmark)
                                print("Centro do Campo está Perto")
                            #Centro do campo longe
                            elif (int(c2_center) <= self.config.y_longe):
                                msg_centerlandmark.far = True
                                self.publisher_centerlandmark.publish(msg_centerlandmark)
                                print("Centro do Campo Longe")
                            #Centro do campo no centro da visão
                            else:
                                msg_centerlandmark.med = True
                                self.publisher_centerlandmark.publish(msg_centerlandmark)
                                print("Centro do Campo está ao Centro")
                                
                        if names[int(cls)] == "penalti":
                            #Lógica para processar a detecção do penalty e publicar posição
                            msg_penaltilandmark.detected = True
                            print("Penalty do campo detectado '%s'" % msg_penaltilandmark.detected)
                            #Penalty do campo à esquerda da visão
                            if (int(c1_center) <= self.config.x_left):
                                msg_penaltilandmark.left = True
                                self.publisher_penaltilandmark.publish(msg_penaltilandmark)
                                print("Penalty do Campo à Esquerda")
                            #Penalty do campo no centro-esquerda da visão
                            elif (int(c1_center) > self.config.x_left and int(c1_center) < self.config.x_center):
                                msg_penaltilandmark.center_left = True
                                self.publisher_penaltilandmark.publish(msg_penaltilandmark)
                                print("Penalty do Campo Centralizado à Esquerda")
                            #Penalty do campo no centro-direita da visão
                            elif (int(c1_center) < self.config.x_right and int(c1_center) > self.config.x_center):
                                msg_penaltilandmark.center_right = True
                                self.publisher_penaltilandmark.publish(msg_penaltilandmark)
                                print("Penalty do Campo Centralizado à a Direita")
                            #Penalty do campo à direita da visão
                            else:
                                msg_penaltilandmark.right = True
                                self.publisher_penaltilandmark.publish(msg_penaltilandmark)
                                print("Penalty do Campo à Direita")
                            #Penalty do campo perto
                            if (int(c2_center) > self.config.y_chute):
                                msg_penaltilandmark.close = True
                                self.publisher_penaltilandmark.publish(msg_penaltilandmark)
                                print("Penalty do Campo está Perto")
                            #Penalty do campo longe
                            elif (int(c2_center) <= self.config.y_longe):
                                msg_penaltilandmark.far = True
                                self.publisher_penaltilandmark.publish(msg_penaltilandmark)
                                print("Penalty do Campo está Longe")
                            #Penalty do campo no centro da visão
                            else:
                                msg_penaltilandmark.med = True
                                self.publisher_penaltilandmark.publish(msg_penaltilandmark)
                                print("Penalty do Campo está ao Centro")
                            
                        if names[int(cls)] == "goalpost":
                            #Lógica para processar a detecção dos goalposts e publicar posição
                            msg_goalpostlandmark.detected = True
                            print("Goalpost detectado '%s'" % msg_goalpostlandmark.detected)
                            #Goalpost à esquerda da visão
                            if (int(c1_center) <= self.config.x_left):
                                msg_goalpostlandmark.left = True
                                self.publisher_goalpostlandmark.publish(msg_goalpostlandmark)
                                print("Goalpost à Esquerda")
                            #Goalpost no centro-esquerda da visão
                            elif (int(c1_center) > self.config.x_left and int(c1_center) < self.config.x_center):
                                msg_goalpostlandmark.center_left = True
                                self.publisher_goalpostlandmark.publish(msg_goalpostlandmark)
                                print("Goalpost Centralizado à Esquerda")
                            #Goalpost no centro-direita da visão
                            elif (int(c1_center) < self.config.x_right and int(c1_center) > self.config.x_center):
                                msg_goalpostlandmark.center_right = True
                                self.publisher_goalpostlandmark.publish(msg_goalpostlandmark)
                                print("Goalpost Centralizado à a Direita")
                            #Goalpost à direita da visão
                            else:
                                msg_goalpostlandmark.right = True
                                self.publisher_goalpostlandmark.publish(msg_goalpostlandmark)
                                print("Goalpost à Direita")
                            #Goalpost perto
                            if (int(c2_center) > self.config.y_chute):
                                msg_goalpostlandmark.close = True
                                self.publisher_goalpostlandmark.publish(msg_goalpostlandmark)
                                print("Goalpost está Perto")
                            #Goalpost longe
                            elif (int(c2_center) <= self.config.y_longe):
                                msg_goalpostlandmark.far = True
                                self.publisher_goalpostlandmark.publish(msg_goalpostlandmark)
                                print("Goalpost está Longe")
                            #Goalpost no centro da visão
                            else:
                                msg_goalpostlandmark.med = True
                                self.publisher_goalpostlandmark.publish(msg_goalpostlandmark)
                                print("Goalpost está ao Centro")
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
            cv2.imshow('Landmark Detection', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#def motor_angle(x):
#    return (((x-1024)*90)/1024)

#def calculate_distance(angle):
#    camera_height = 0.5  #Altura do motor do pescoço até a câmera em metros
#    robot_height = 2  #Altura do robô (até o pescoço)
#    angle_rad = math.radians(angle) #Ângulo em radianos

#    y = camera_height * math.sin(angle_rad)
#    x = camera_height * math.cos(angle_rad)

#    total_height = high + y
#    distance = math.tan(angle_rad) * total_height + x
#    return distance
    

def main(args=None):
    rclpy.init(args=args)
    config = classConfig()
    landmark_detection = LandmarkDetection(config)
    rclpy.spin(landmark_detection)
    landmark_detection.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()