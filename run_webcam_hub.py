import logging 
import sys
from signalrcore.hub_connection_builder import HubConnectionBuilder
import argparse
import time
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def RecieveConnId(message):
    logging.info(message)
def RecieveMessage(username, message):
    logging.info(f"username:{username}, message:{message}")

def input_with_default(input_text, default_value):
    value = input(input_text.format(default_value))
    return default_value if value is None or value.strip() == "" else value

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
fps_time = 0

if __name__ == "__main__":
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
    parser.add_argument('--server_url', type=str, default="http://10.0.0.45:5000/chatHub",
                        help="url for signal r hub")
    parser.add_argument('--username', type=str, default="aicamsys",
                        help="username shown in signal r hub")                                       
    args = parser.parse_args()

    #server_url = input_with_default("Enter your server url(default: {0}): ","http://10.0.0.45:5000/chatHub")
    #username = input_with_default("Enter your username (default: {0}): ", "arifsohaib")
    server_url = args.server_url
    username = args.username

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    hub_connection = HubConnectionBuilder().with_url(server_url, options={"verify_ssl":False})\
        .configure_logging(logging.DEBUG, socket_trace=True, handler=handler)\
        .with_automatic_reconnect({"type":"interval", "keep_alive_interval":10, "intervals":[1,3,5,7]}).build()

    hub_connection.on_open(lambda:print("connection opened and handshake recieved readt to send messages"))
    hub_connection.on_close(lambda:print("connection closed"))

    hub_connection.on("RecieveMessage", RecieveMessage)
    hub_connection.on("RecieveConnId", RecieveConnId)

    hub_connection.start()


    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    logger.debug("cam read+")
    dispW = 640
    dispH = 480
    flip = 2
    camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true'
    cam = cv2.VideoCapture(camSet)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    message = None
    while True:
        ret_val, image = cam.read()
        logger.debug("image process +")
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        logger.debug(str(humans))
        logger.debug('postprocess+')
        image,pairs, angles = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        if(angles):
            thetas = []
            for tup in angles:
                if ((tup[1][0]-tup[0][0]) != 0) and ((tup[2][0]-tup[1][0]) != 0):
                    slope1 = (tup[1][1]-tup[0][1])/(tup[1][0]-tup[0][0])
                    slope2 = (tup[2][1]-tup[1][1])/(tup[2][0]-tup[1][0])
                    if (1 + (slope2 *slope1)) !=0:
                        tanTheta = np.abs((slope2 - slope1)/(1 + (slope2 *slope1)))
                        theta = np.arctan(tanTheta)*57.2958
                        cv2.putText(image, str(np.round(theta,2)), (tup[1][0]+5,tup[1][1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                        thetas.append(theta)
            anglesSTR = f"angles: {str(angles)} thetas: {str(thetas)}"
            
            hub_connection.send("SendMessage", [username, anglesSTR])
        logger.debug("show+")
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()                
        if cv2.waitKey(1) == 27:
            break
        logger.debug("finished+")
    cv2.destroyAllWindows()
    hub_connection.stop()
    sys.exit()

    # while message != "exit()":
    #     message = input(">>")
    #     if message is not None and message != "" and message != "exit()":
    #         hub_connection.send("SendMessage", [username, message])
    
    

