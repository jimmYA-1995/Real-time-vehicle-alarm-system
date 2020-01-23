from ctypes import *
import time
import glob
import cv2
import csv
import json
import numpy as np
# import pyautogui as pag
from kafka import KafkaProducer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder(self, obj)
    


def set_gaze_estimation_lib():
    for lib in glob.glob("/usr/local/lib/libopencv*.so"):
        CDLL(lib, mode= RTLD_GLOBAL)
    libgaze = CDLL("libgaze_estimation_demo.so")
    libgaze.get_gaze_x_py.restype = c_float
    libgaze.get_gaze_y_py.restype = c_float
    libgaze.get_gaze_z_py.restype = c_float
    return libgaze

if __name__ == "__main__":
    kafka_producer = KafkaProducer(bootstrap_servers='192.168.50.59',
                                   value_serializer=lambda v: json.dumps(v, cls=NumpyEncoder).encode('utf-8'),
                                   api_version=(2,4,0))
    
    # cv2.namedWindow("WindowImg")
    screenSize = (1920, 1080) # pag.size()
    circleRadius = 10
    circleColor = (0,0,255)


    libgaze = set_gaze_estimation_lib()
    gazeclass = libgaze.gazeClass_py()
    poly = PolynomialFeatures(2,include_bias=True, interaction_only=False)
    poly.fit(np.array([[0.5,0.5,0.5]]))
    regX = LinearRegression()
    regY = LinearRegression()
    itemX = None
    biasX = 0.0
    with open("test.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                itemX = [float(a) for a in row[0:-1]]
                itemX = np.array(itemX)
                biasX = float(row[-1])
            if line_count == 1:
                itemY = [float(a) for a in row[0:-1]]
                itemY = np.array(itemY)
                biasY = float(row[-1])
            line_count += 1
    regX.coef_ = itemX
    regY.coef_ = itemY
    regX.intercept_ = biasX
    regY.intercept_ = biasY
    #i = 0
    #time_acc = 0.
    #kafka_acc = 0.
    #total_s = time.time()
    while True:
        # i += 1
        # start = time.time()
        libgaze.exeEsti_py(gazeclass)
        gazeVec = [libgaze.get_gaze_x_py(gazeclass),libgaze.get_gaze_y_py(gazeclass),libgaze.get_gaze_z_py(gazeclass)]
        tmpX = regX.predict(poly.transform([gazeVec]))
        tmpY = regY.predict(poly.transform([gazeVec]))
        tmpX = np.clip(tmpX,0,screenSize[0])
        tmpY = np.clip(tmpY,0,screenSize[1])
        # time_acc += (time.time() - start)
        # screenImg = np.zeros((screenSize[1],screenSize[0],3), np.uint8)
        # screenImg.fill(255)
        # screenImg = cv2.circle(screenImg, (tmpX,tmpY), circleRadius, circleColor, -1)
        # cv2.imshow("WindowImg", screenImg)
        print("x: {}, y: {}".format(tmpX[0],tmpY[0]))
        # start = time.time()
        kafka_producer.send('gaze', {'coordinate': (tmpX[0], tmpY[0])})
        # kafka_acc += (time.time() - start)
        # tmpKb = cv2.waitKey(1)
        # q = 113, s = 115, space = 32
        #if tmpKb == 113:
         #   break

        # if i == 1000:
        #    break
   # print("total: ", time.time() - total_s)
   # print("average gaze alg: ", time_acc / 1000)
   # print("average kafka: ", kafka_acc / 1000)
