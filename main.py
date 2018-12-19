import cv2
import time
import imutils
import logging
import argparse
import datetime
import numpy as np
import tensorflow as tf

from utils import detector_utils as detector_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('detector')
detection_graph, sess = detector_utils.load_inference_graph()

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-wd',
    '--width',
    dest='width',
    type=int,
    default=1080,
    help='Width of the frames in the video stream.')
parser.add_argument(
    '-ht',
    '--height',
    dest='height',
    type=int,
    default=720,
    help='Height of the frames in the video stream.')
parser.add_argument(
    '-x1_ROI',
    '--x1_ROI',
    dest='x1_ROI',
    type=int,
    default=580,
    help='ROI of detection p1_x.')
parser.add_argument(
    '-x2_ROI',
    '--x2_ROI',
    dest='x2_ROI',
    type=int,
    default=780,
    help='ROI of detection p2_x.')
parser.add_argument(
    '-y1_ROI',
    '--y1_ROI',
    dest='y1_ROI',
    type=int,
    default=500,
    help='ROI of detection p1_y.')
parser.add_argument(
    '-y2_ROI',
    '--y2_ROI',
    dest='y2_ROI',
    type=int,
    default=720,
    help='ROI of detection p2_y.')
parser.add_argument(
    '-x1_CB',
    '--x1_CB',
    dest='x1_CB',
    type=int,
    default=575,
    help='ROI of cashbox p1_x.')
parser.add_argument(
    '-x2_CB',
    '--x2_CB',
    dest='x2_CB',
    type=int,
    default=707,
    help='ROI of cashbox p2_x.')
parser.add_argument(
    '-y1_CB',
    '--y1_CB',
    dest='y1_CB',
    type=int,
    default=582,
    help='ROI of cashbox p1_y.')
parser.add_argument(
    '-y2_CB',
    '--y2_CB',
    dest='y2_CB',
    type=int,
    default=661,
    help='ROI of cashbox p2_y.')
parser.add_argument(
    '-sth',
    '--scorethreshold',
    dest='score_thresh',
    type=float,
    default=0.5,
    help='Score threshold for displaying bounding boxes')
parser.add_argument(
    '-th',
    '--threshold',
    dest='threshold',
    type=int,
    default=3,
    help='Threshold value for cashbox detection')
parser.add_argument(
    '-LB',
    '--LowerB',
    dest='LOWERB',
    nargs='+',
    type=int,
    default=[100, 0, 0],
    help='Lower Boundary colour value for cashbox detection')
parser.add_argument(
    '-UB',
    '--UpperB',
    dest='UPPERB',
    nargs='+',
    type=int,
    default=[200, 80, 80],
    help='Upper Boundary colour value for cashbox detection')
parser.add_argument(
    '-src',
    '--source',
    dest='video',
    type=str,
    default=r'D:\Lauretta\Storehub\footage1_Trim.mp4',
    help='Path to input video file.')
parser.add_argument(
    '-hd',
    '--hand(s)',
    dest='num_hands_detect',
    type=int,
    default=2,
    help='Max number of hands we want to detect/track.')
args = parser.parse_args()

cap = cv2.VideoCapture(r'D:\Lauretta\Storehub\transaction-2018-11-22-16-08-45.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Variables
im_width, im_height = (args.width, args.height)
start = time.time()
t = time.time()
statusInt = -1
currentStatus = -1
transactionCount = 0
recordOnce = True
lowerStatus = True
lowestStatus = True
handUp = False
record = False
boxOpen = False
startCount = False
isRecodring = False
nearlyEndedTransaction = False
arr = [True, False, False, False]
out = None

tnow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
out = cv2.VideoWriter('transaction-' + tnow + '.avi',
                      fourcc, 30.0, (640, 360))

##################################### MAIN #####################################
while True:
    ret, image_np = cap.read()
    ori = image_np.copy()
    image_np = cv2.resize(image_np, (args.width, args.height))
    frameCB = image_np[args.y1_CB:args.y2_CB, args.x1_CB:args.x2_CB]
    maskCB = cv2.inRange(frameCB, np.array(args.LOWERB), np.array(args.UPPERB))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np1 = np.zeros_like(image_np)
    image_np1[args.y1_ROI:args.y2_ROI,
              args.x1_ROI:args.x2_ROI] = image_np[args.y1_ROI:args.y2_ROI, args.x1_ROI:args.x2_ROI]
    boxes, scores = detector_utils.detect_objects(image_np1,
                                                  detection_graph, sess)

    condCB = np.count_nonzero(maskCB) > args.threshold
    condHand = any([s > args.score_thresh for s in scores])

    # Cashbox CLOSED
    if nearlyEndedTransaction & (np.count_nonzero(maskCB) < args.threshold):
        start = time.time()
        status = 'CASH BOX CLOSED!'
        statusInt = 3
        arr[3] = True
        startCount = False
        lowerStatus = False
    else:
        arr[3] = False

    # Cashbox OPENED & hand DETECTED
    if boxOpen & condCB & condHand:
        if lowerStatus:
            status = 'HAND DETECTED!'
            statusInt = 2
        for i in range(args.num_hands_detect):
            detector_utils.draw_box_on_image(i, boxes, im_width,
                                             im_height, image_np)
        arr[2] = True
        handUp = True
        record = True
        startCount = False
        lowestStatus = False
        nearlyEndedTransaction = True
    else:
        arr[2] = False
        handUp = False

    # Cashbox OPENED
    if condCB:
        if not(startCount):
            start = time.time()
            startCount = True
        if lowestStatus:
            status = 'CASH BOX OPENED!'
            statusInt = 1
        arr[1] = True
        #record = True
        boxOpen = True
    else:
        arr[1] = False
        boxOpen = False

    # No event happened
    if arr == [True, False, False, False]:
        status = 'None'
        statusInt = 0

    elif arr == [True, False, False, True]:
        arr = [True]*4

    # Transaction COMPLETED
    if statusInt != currentStatus:
        currentStatus = statusInt
        logger.info(tnow + ' ' + status + ' ref: %s, STATUS: %s',
                    np.count_nonzero(maskCB), [x*1 for x in arr])
        if statusInt == 3:
            record = False
            lowerStatus = True
            lowestStatus = True
            nearlyEndedTransaction = False
            arr = [True, False, False, False]
            transactionCount += 1
            logger.info('-----------------------------------------')
            logger.info('----------TRANSACTION ENDED--------------')
            logger.info('-----------------------------------------')

    cv2.putText(ori, 'Transaction: ' + str(transactionCount), (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(ori, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Video Recording
    if record:
        out.write(ori)

    # Warning if cashbox OPENED for more than 50 seconds
    if startCount and (time.time() - start > 50):
        logger.warning('Cash box opened for more than 50 seconds!')
        record = False

    # cv2.rectangle(image_np, (args.x1_ROI,args.y1_ROI),
    #              (args.x2_ROI,args.y2_ROI), (255, 0, 0), 3, 1)

    cv2.imshow('Video Stream', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
