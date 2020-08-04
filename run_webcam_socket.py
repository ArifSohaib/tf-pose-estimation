import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import socket
import sys

HEADER_LENGTH = 10

IP = "10.0.0.45"
PORT = 1234
my_username = "ai_cam_sys"
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP,PORT))
client_socket.setblocking(False)


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0



#begin to send info to server
username = my_username.encode('utf-8')
username_header = f"{len(username):<{HEADER_LENGTH}}".encode("utf-8")
client_socket.send(username_header + username)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='256x256',
                        help='if provided, resize images before they are processed. default=256x256, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    dispW = 640
    dispH = 480
    flip=2
    camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true'

    cam = cv2.VideoCapture(camSet)
    #cam = cv2.VideoCapture(args.camera)
    #cam = cv2.VideoCapture("/home/aicamsys/family_crossing.mp4")
    ret_val, image = cam.read()
    
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    
    while True:
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        logger.debug(str(humans))
        # if(humans):
        #     message = str(humans).encode("utf-8")
        #     message_header = f"{len(message):<{HEADER_LENGTH}}".encode("utf-8")
        #     client_socket.send(message_header + message)
        
        logger.debug('postprocess+')
        image,pairs, angles = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # if(pairs):
        #     pairs = f"pairs: {str(pairs)}".encode("utf-8")
        #     message_header = f"{len(pairs):<{HEADER_LENGTH}}".encode("utf-8")
        #     client_socket.send(message_header + pairs)
        if(angles):
            thetas = []
            for tup in angles:
                "angles contain tuple of 3 with x,y values in it"
                if ((tup[1][0]-tup[0][0]) != 0) and ((tup[2][0]-tup[1][0]) != 0):

                    slope1 = (tup[1][1]-tup[0][1])/(tup[1][0]-tup[0][0])
                    slope2 = (tup[2][1]-tup[1][1])/(tup[2][0]-tup[1][0])
                    if (1 + (slope2 *slope1)) !=0:
                    
                        tanTheta = np.abs((slope2 - slope1)/(1 + (slope2 *slope1)))
                        theta = np.arctan(tanTheta)*57.2958
                        cv2.putText(image, str(np.round(theta,2)), (tup[1][0]+5,tup[1][1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                        thetas.append(theta)
                #logger.debug(f"angle points:\n{str(angles)},angle:\n{thetas}")
            anglesSTR = f"angles: {str(angles)} thetas: {str(thetas)}".encode("utf-8")
            message_header = f"{len(anglesSTR):<{HEADER_LENGTH}}".encode("utf-8")
            client_socket.send(message_header + anglesSTR)
        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
