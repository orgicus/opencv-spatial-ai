#!/usr/bin/env python3
import argparse
import cv2
import depthai as dai
import blobconverter
import numpy as np
from depthai_sdk import Replay, Previews, FPSHandler
from depthai_sdk.utils import frameNorm, cropToAspectRatio
from depthai_sdk.managers import PipelineManager, PreviewManager, BlobManager, NNetManager


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
args = parser.parse_args()

# Create Replay objects
replay = Replay(args.path)

model_path  = "yolov7EscalatorsTiny/yolov7escalatorstiny_openvino_2021.4_6shave.blob"
config_path = "yolov7EscalatorsTiny/yolov7escalatorstiny.json"
w, h = 640, 640

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

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(w, h)
manip.setMaxOutputFrameSize(w * h * 3)
nodes.color.out.link(manip.inputImage)

# create yolo node
bm = BlobManager(blobPath=model_path)
nm = NNetManager(inputSize=(w, h), nnFamily="YOLO")
nm.readConfig(config_path)
nn = nm.createNN(pipeline=pipeline, nodes=nodes, source=Previews.color.name,
                 blobPath=bm.getBlob(shaves=6, openvinoVersion=pipeline.getOpenVINOVersion()))

# # Link required inputs to the Spatial detection network
manip.out.link(nn.input)
nodes.stereo.depth.link(nn.inputDepth)

detOut = pipeline.create(dai.node.XLinkOut)
detOut.setStreamName("det_out")
nn.out.link(detOut.input)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth_out")
nodes.stereo.disparity.link(depthOut.input)

with dai.Device(pipeline) as device:
    fpsHandler = FPSHandler()
    replay.createQueues(device)

    pv = PreviewManager(display=[Previews.color.name], fpsHandler=fpsHandler)

    depthQ = device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
    nm.createQueues(device)
    
    nnData = []

    disparityMultiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()
    color = (255, 0, 0)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    while replay.sendFrames():
        rgbFrame = cropToAspectRatio(replay.frames['color'], (w, h))

        depthFrame = depthQ.get().getFrame()
        depthFrameColor = (depthFrame*disparityMultiplier).astype(np.uint8)
        # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        # inNn = nm.outputQueue.tryGet()

        # if inNn is not None:
        #     # count FPS
        #     fpsHandler.tick("color")

        #     nnData = nm.decode(inNn)

        # nm.draw(pv, nnData)
        # pv.showFrames()

        
        # cv2.imshow("rgb", rgbFrame)
        # cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break
    print('End of the recording')
