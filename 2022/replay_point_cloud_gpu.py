#!/usr/bin/env python3
import os
import traceback
import argparse
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import Replay
from depthai_sdk.utils import frameNorm, cropToAspectRatio

from projector_3d_gpu import PointCloudVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
args = parser.parse_args()

w, h = 640, 360

# Create Replay objects
replay = Replay(args.path)

replay.disableStream('depth') # In case depth was saved (mcap)
# Resize color frames prior to sending them to the device
replay.setResizeColor((w, h))
# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case 300x300)
replay.keepAspectRatio(True)

# Initializes the pipeline. This will create required XLinkIn's and connect them together
# Creates StereoDepth node, if both left and right streams are recorded
pipeline, nodes = replay.initPipeline()

nodes.stereo.setSubpixel(True)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth_out")
nodes.stereo.disparity.link(depthOut.input)

with dai.Device(pipeline) as device:
    # setup point cloud visualisation
    # calibData = device.readCalibration()
    calibJsonFile = os.path.join(args.path, 'calib.json')
    calibData = dai.CalibrationHandler(calibJsonFile)
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(w, h))
    print("self.intrinsics", intrinsics)
    pcl_converter = PointCloudVisualizer(intrinsics, w, h)
    
    replay.createQueues(device)
    
    depthQ = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
    
    disparityMultiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()
    # color = (255, 0, 0)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.sendFrames():
        # rgbFrame = cropToAspectRatio(replay.frames['color'], (300,300))
        rgbFrame = cv2.resize(replay.frames['color'], (w, h))
        depthFrame = depthQ.get().getFrame()
        depthFrameColor = (depthFrame*disparityMultiplier).astype(np.uint8)
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        
        cv2.imshow("rgb", rgbFrame)
        cv2.imshow("depth", depthFrameColor)

        pcl_converter.rgbd_to_projection(depthFrame, cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2RGB), downsample=True, remove_noise=True)
        pcl_converter.visualize_pcd()

        if cv2.waitKey(1) == ord('q'):
            break
    print('End of the recording')

