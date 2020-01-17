#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import time
import json
import argparse
from glob import glob
from multiprocessing import Process, Event, Queue

import cv2
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

from trackiou import Tracker
from kafka import KafkaConsumer # KafkaProducer, 
from edgetpu.detection.engine import DetectionEngine


class NumpyEncoder(json.JSONEncoder):
    """ For serialize numpy object. """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Consumer(Process):
    def __init__(self):
        super().__init__()
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        consumer = KafkaConsumer('gaze',
                                 bootstrap_servers='192.168.50.170:9092',
                                 value_deserializer=lambda v: json.loads(v, encoding='utf-8'),
                                 api_version=(2,4,0))
        while not self.stop_event.is_set():
            for msg in consumer:
                print("get result: {}, {}".format(msg.value['x'], msg.value['y']))
                gaze_queue.put((msg.value['x'], msg.value['y']))

                if self.stop_event.is_set():
                    break
        consumer.close()

def make_interpreter(model_file):
    EDGETPU_SHARED_LIB = '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1'
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB)])

def parse_label_file(args):
    print("[INFO] parsing class labels...")
    labels = {}
    for row in open(args["labels"]):
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()
    
    return labels


def draw_bbox_and_show(orig, tracks, timestr, i, gaze_data=None, store=False):
    # draw gaze data as red dot
    if gaze_data is not None:
        print("printing circle...")
        for x, y in gaze_data:
            print(x, y)
            cv2.circle(orig, (int(x), int(y)), 3, (0, 0, 255), -1)

    # decide the color of BBox depending on noticed & dangerous
    for r in tracks:
        (startX, startY, endX, endY) = r['bbox']
        label = r['label']
        track_id = r['trackid']
        if r['noticed'] and r['dangerous']:
            cmap = (0, 140, 255)
        elif r['noticed']:
            cmap = (255, 0 , 0)
        elif r['dangerous']:
            cmap = (0, 0, 255)
        else: cmap = (0, 255, 0)
            
        cv2.rectangle(orig, (startX, startY), (endX, endY),
            cmap, 2)

        # adjust offset to fit tackID
        y = startY - 15 if startY - 15 > 15 else startY + 15
        text = "{} ID:{}".format(label, str(track_id))

        cv2.putText(orig, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cmap, 1)
    
    # show the output frame and wait for a key press
    cv2.imshow("Frame", orig)
    if store:
        cv2.imwrite("/sd_mnt/data/{}/track/{}.png".format(timestr, i), orig)
    return cv2.waitKey(1) & 0xFF


def main(args):
    labels = parse_label_file(args)

    print("[INFO] loading detection model")
    model = DetectionEngine(args["model"])

    print("[INFO] make collide interpreter")
    collide_interpreter = make_interpreter(args['collide_model'])
    collide_interpreter.allocate_tensors()

    print("[INFO] create tracker")
    tracker = Tracker(args, collide_interpreter)

    print("[INFO] Initialize Kafka worker")
    # producer = KafkaProducer(bootstrap_servers='192.168.101.82:9092',
    #                         value_serializer=lambda v: json.dumps(v, cls=NumpyEncoder).encode('utf-8'),
    #                         api_version=(2,4,0))
    consumer =  KafkaConsumer('gaze',
                              bootstrap_servers='192.168.50.59:9092',
                              value_deserializer=lambda v: json.loads(v , encoding='utf-8'),
                              max_poll_records=30,
                              api_version=(2,4,0))

    print("[INFO] Warmup FrameGrabber...")
    if args['streaming']:
        vs = cv2.VideoCapture(1)
        vs.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 375)
    else:
        video_dir = input("enter the timetamp of target directory: ")
        total_frames = len(glob("/sd_mnt/data/{}/origin/*.png".format(timestamp)))
    
    # initialize the video stream and allow the camera sensor to warmup
    time.sleep(2.0)

    read_acc = []
    proprocess_acc = []
    model_acc = []
    track_acc = []
    fps_acc = []

    timestr = time.strftime("%m%d-%H%M%S")
    if not os.path.exists('/sd_mnt/data/{}/origin'.format(timestr)):
        os.makedirs('/sd_mnt/data/{}/origin'.format(timestr))
    if not os.path.exists('/sd_mnt/data/{}/track'.format(timestr)):
        os.makedirs('/sd_mnt/data/{}/track'.format(timestr))

    i=0
    while True:
        # only track time after 200th frames to avoid warmup
        s = time.time()

        if args['streaming']:
            _, frame = vs.read()
        else:
            frame = cv2.imread('/sd_mnt/data/{}/origin/{}.png'.format(video_dir, str(i).zfill(4)))

        gaze_points = None
        rescale = np.array([500/1920, 375/1080])
        d = consumer.poll()
        tmp = [msg.value for msgs in d.values() for msg in msgs]
        if tmp:
            gaze_points = [data['coordinate'] for data in tmp]
            gaze_points = np.array(gaze_points)
            gaze_points *= rescale
            print(gaze_points)
        else:
            print("----- no gaze data -------")

        read_acc.append(time.time() - s)

        preprocess_begin = time.time()
        print("&" * 30, "resize: ", time.time() - preprocess_begin)
        orig = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        proprocess_acc.append(time.time() - preprocess_begin)

        # make predictions on the input frame
        detect_begin = time.time()
        results = model.detect_with_image(frame, threshold=args["confidence"], keep_aspect_ratio=True, relative_coord=False)
        model_acc.append(time.time() - detect_begin)

        detections = []
        for r in results:
            if r.label_id < 8 and r.label_id != 4:
                if r.label_id in [2,5,6]:
                    detections.append({'bbox': r.bounding_box.flatten().astype("int"), 'label': 2, 'score': str(r.score)})
                else:
                    detections.append({'bbox': r.bounding_box.flatten().astype("int"), 'label': r.label_id, 'score': str(r.score)})

        track_begin = time.time()
        tracker.track(detections, gaze_points)
        # producer.send('bbox', {'bboxes': detections})
        track_acc.append(time.time() - track_begin)

        key = draw_bbox_and_show(orig, tracker.get_current(), timestr, i, gaze_points)
        
        e = time.time()
        fps_acc.append(1 / (e - s))

        i += 1
        if key == ord("q"):
            break
        elif not args['streaming'] and i == total_frames:
            break

    cv2.destroyAllWindows()

    read_acc = np.array(read_acc)
    proprocess_acc = np.array(proprocess_acc)    
    model_acc = np.array(model_acc)
    track_acc = np.array(track_acc)
    fps_acc = np.array(fps_acc)

    print("read_acc: ", read_acc.mean(), read_acc.max(), read_acc.min())
    print("proprocess_acc: ", proprocess_acc.mean(), proprocess_acc.max(), proprocess_acc.min())
    print("model_acc: ", model_acc.mean(), model_acc.max(), model_acc.min())
    print("track_acc: ", track_acc.mean(), track_acc.max(), track_acc.min())
    print("fps_acc: ", fps_acc.mean(), fps_acc.max(), fps_acc.min())    


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--streaming", action="store_true",
                    help="whether use streaming or take from picture")
    ap.add_argument("--collide_model", required=False,
        default="/home/mendel/models/collide_softmax/no_quantized.tflite",
        help="path to TensorFlow Lite collide prediction model")
    ap.add_argument("-m", "--model", required=False,
        # default="/home/mendel/models/co_compile/model_postprocessed_quantized_128_uint8_edgetpu.tflite",
        default="/home/mendel/models/mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
        help="path to TensorFlow Lite object detection model")
    ap.add_argument("-l", "--labels", required=False,
        default="/home/mendel/models/mobilenet_ssd_v2/coco_small.txt",
        help="path to labels file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3,
        help="minimum probability to filter weak detections")
    ap.add_argument("-w", "--sliding_window", type=int, default=3,
        help="width of sliding window")
    ap.add_argument("--lifespan", type=int, default=4,
        help="lifespan of tracks")
    ap.add_argument("--sigma_iou", type=float, default=0.6,
        help="threshold of the iou between target object and track")
    args = vars(ap.parse_args())

    main(args)
