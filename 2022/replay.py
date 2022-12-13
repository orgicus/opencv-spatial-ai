#!/usr/bin/env python3
import argparse
import cv2
import depthai as dai
import blobconverter
import numpy as np
from depthai_sdk import Replay
from depthai_sdk.utils import frameNorm, cropToAspectRatio

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
args = parser.parse_args()

# Create Replay objects
replay = Replay(args.path)

replay.disableStream('depth') # In case depth was saved (mcap)
# Resize color frames prior to sending them to the device
replay.setResizeColor((640, 400))
# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case 300x300)
replay.keepAspectRatio(True)

# Initializes the pipeline. This will create required XLinkIn's and connect them together
# Creates StereoDepth node, if both left and right streams are recorded
pipeline, nodes = replay.initPipeline()

nodes.stereo.setSubpixel(True)

# manip = pipeline.create(dai.node.ImageManip)
# manip.initialConfig.setResize(300,300)
# manip.setMaxOutputFrameSize(300*300*3)
# nodes.color.out.link(manip.inputImage)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth_out")
nodes.stereo.disparity.link(depthOut.input)

with dai.Device(pipeline) as device:
    replay.createQueues(device)

    depthQ = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
    
    disparityMultiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()
    # color = (255, 0, 0)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.sendFrames():
        # rgbFrame = cropToAspectRatio(replay.frames['color'], (300,300))
        rgbFrame = cv2.resize(replay.frames['color'], (640, 400))
        depthFrame = depthQ.get().getFrame()
        depthFrameColor = (depthFrame*disparityMultiplier).astype(np.uint8)
        # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
        print(rgbFrame.shape)
        cv2.imshow("rgb", rgbFrame)
        cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break
    print('End of the recording')

