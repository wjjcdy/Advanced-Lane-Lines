# -*- coding: UTF-8 -*-
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
dirs = os.listdir("camera_cal/")
# prepare object points
NX = 9 #TODO: enter the number of inside corners in x
NY = 6 #TODO: enter the number of inside corners in y

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
   # undist = np.copy(img)  # Delete this line
    return dst

def write_video_test():
    cap = cv2.VideoCapture('project_video.mp4')#打开相机
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fps = int(cap.get(5))
    print(fps)
    codec = int(cap.get(6))
    print(codec)
    print(cv2.CAP_PROP_FOURCC)
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    print ('codec is %x'%(codec))
    print ('codec is ' + chr(codec&0xFF) + chr((codec>>8)&0xFF) + chr((codec>>16)&0xFF) + chr((codec>>24)&0xFF))

    out = cv2.VideoWriter('output_images/project_video.mp4', int(codec), int(fps), (frame_width,frame_height),1)
    while(True):
        ret,frame = cap.read()#捕获一帧图像
        out.write(frame)#保存帧
        cv2.imshow('frame',frame)#显示帧
        #判断按键，如果按键为q，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()#关闭相机
    out.release()
    cv2.destroyAllWindows()

def main():
    print(__file__ + " start!!")
    print(" start!!")
    objpoints = []                                #真实环境对应点组
    imgpoints = []                                #对应图片中的点组
    objp = np.zeros((NX*NY,3), np.float32)        #根据提供的棋盘格的个数分布
    objp[:,:2] = np.mgrid[0:NX,0:NY].T.reshape(-1,2)

    for file in dirs:                             #遍历参考的所有图片
        img = cv2.imread("camera_cal/"+file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
        # If found, draw corners
        if ret == True:
           imgpoints.append(corners)
           objpoints.append(objp)
           cv2.drawChessboardCorners(img, (NX, NY), corners, ret)

    img_test1 = cv2.imread("test_images/straight_lines1.jpg")
    img_test1 = cv2.cvtColor(img_test1, cv2.COLOR_BGR2RGB)
    #dst = cal_undistort(img_test1, objpoints, imgpoints)
    plt.imshow(img_test1)
    plt.show()
    print("end")

if __name__ == '__main__':
    # main()
    write_video_test()
