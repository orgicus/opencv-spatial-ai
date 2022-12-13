#!/usr/bin/env python3

import os, argparse

import cv2
import numpy as np

import depthai as dai
from depthai_sdk import Replay
from depthai_sdk.utils import cropToAspectRatio

class OAK:

    COLOR = True

    lrcheck  = True   # Better handling for occlusions
    extended = False  # Closer-in minimum depth, disparity range is doubled
    subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
    median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

    def __init__(self, path_to_recording: str) -> None:
        print("StereoDepth config options:")
        print("    Left-Right check:  ", self.lrcheck)
        print("    Extended disparity:", self.extended)
        print("    Subpixel:          ", self.subpixel)
        print("    Median filtering:  ", self.median)

        self.is_live = not os.path.exists(path_to_recording)

        if not self.is_live:
            self.replay = Replay(path_to_recording)
            self.replay.disableStream('depth') # In case depth was saved (mcap)
            # Resize color frames prior to sending them to the device
            self.w, self.h = (640, 480)
            self.replay.setResizeColor((self.w, self.h))
            # Keep aspect ratio when resizing the color frames. This will crop
            # the color frame to the desired aspect ratio (in our case 300x300)
            self.replay.keepAspectRatio(True)
            self.pipeline, self.nodes = self.replay.initPipeline()
            self.depthOut = self.pipeline.create(dai.node.XLinkOut)
            self.depthOut.setStreamName("depth_out")
            self.nodes.stereo.disparity.link(self.depthOut.input)
            
            self.setup_device_replay()
        else:

            self.pipeline = dai.Pipeline()

            self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
            self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            self.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

            self.monoRight = self.pipeline.create(dai.node.MonoCamera)
            self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            self.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

            self.stereo = self.pipeline.createStereoDepth()
            self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            self.stereo.initialConfig.setMedianFilter(self.median)
            # stereo.initialConfig.setConfidenceThreshold(255)

            self.stereo.setLeftRightCheck(self.lrcheck)
            self.stereo.setExtendedDisparity(self.extended)
            self.stereo.setSubpixel(self.subpixel)
            self.monoLeft.out.link(self.stereo.left)
            self.monoRight.out.link(self.stereo.right)

            self.config = self.stereo.initialConfig.get()
            self.config.postProcessing.speckleFilter.enable = False
            self.config.postProcessing.speckleFilter.speckleRange = 50
            self.config.postProcessing.temporalFilter.enable = True
            self.config.postProcessing.spatialFilter.enable = True
            self.config.postProcessing.spatialFilter.holeFillingRadius = 2
            self.config.postProcessing.spatialFilter.numIterations = 1
            self.config.postProcessing.thresholdFilter.minRange = 400
            self.config.postProcessing.thresholdFilter.maxRange = 200000
            self.config.postProcessing.decimationFilter.decimationFactor = 1
            self.stereo.initialConfig.set(self.config)

            self.xout_depth = self.pipeline.createXLinkOut()
            self.xout_depth.setStreamName('depth')
            self.stereo.depth.link(self.xout_depth.input)

            # xout_disparity = pipeline.createXLinkOut()
            # xout_disparity.setStreamName('disparity')
            # stereo.disparity.link(xout_disparity.input)

            self.xout_colorize = self.pipeline.createXLinkOut()
            self.xout_colorize.setStreamName('colorize')
            if self.COLOR:
                self.camRgb = self.pipeline.create(dai.node.ColorCamera)
                self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
                self.camRgb.setIspScale(1, 3)
                self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
                self.camRgb.initialControl.setManualFocus(130)
                self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
                self.camRgb.isp.link(self.xout_colorize.input)
            else:
                self.stereo.rectifiedRight.link(self.xout_colorize.input)

            self.setup_device_live()

        self.on_rgbd_frame = None

    def setup_device_live(self):
        self.device = dai.Device(self.pipeline)

        self.qs = []
        self.qs.append(self.device.getOutputQueue("depth", maxSize=1, blocking=False))
        self.qs.append(self.device.getOutputQueue("colorize", maxSize=1, blocking=False))

        self.calibData = self.device.readCalibration()
        if self.COLOR:
            self.w, self.h = self.camRgb.getIspSize()
            self.intrinsics = self.calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(self.w, self.h))
        else:
            self.w, self.h = self.monoRight.getResolutionSize()
            self.intrinsics = self.calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(self.w, self.h))

        self.sync = HostSync()

    def setup_device_replay(self):
        self.device = dai.Device(self.pipeline)

        self.replay.createQueues(self.device)

        self.depthQ = self.device.getOutputQueue(name="depth_out", maxSize=4, blocking=False)
        
        self.disparityMultiplier = 255 / self.nodes.stereo.initialConfig.getMaxDisparity()

        self.calibData = self.device.readCalibration()
        self.intrinsics = self.calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(self.w, self.h))
        print("self.intrinsics", self.intrinsics)

    def update(self):
        if self.is_live:
            self.update_live()
        else:
            self.update_replay()

    def update_live(self):
        for q in self.qs:
            new_msg = q.tryGet()
            if new_msg is not None:
                msgs = self.sync.add_msg(q.getName(), new_msg)
                if msgs:
                    depth = msgs["depth"].getFrame()
                    color = msgs["colorize"].getCvFrame()
                    
                    if self.on_rgbd_frame is not None:
                        self.on_rgbd_frame(depth, color)

    def update_replay(self):
        has_frames = self.replay.sendFrames()
        if has_frames:
            # color = cropToAspectRatio(self.replay.frames['color'], (300,300))
            color = self.replay.frames['color']
            color = cv2.resize(color, (self.w, self.h))
            depth = self.depthQ.get().getFrame()
            # print('depth', depth.shape, 'color', color.shape)
            # depthFrameColor = (depth * self.disparityMultiplier).astype(np.uint8)
            # # depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            # # depthFrameColor = cv2.equalizeHist(depthFrameColor)
            # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            if self.on_rgbd_frame is not None:
                self.on_rgbd_frame(depth, color)



class HostSync:
    def __init__(self):
        self.arrays = {}
    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({'msg': msg, 'seq': msg.getSequenceNum()})

        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                if msg.getSequenceNum() == obj['seq']:
                    synced[name] = obj['msg']
                    break
        # If there are 3 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 2: # color, depth, nn
            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if obj['seq'] < msg.getSequenceNum():
                        arr.remove(obj)
                    else: break
            return synced
        return False

if __name__ == '__main__':

    from projector_3d import PointCloudVisualizer

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default="data", type=str, help="Path where to store the captured data")
    args = parser.parse_args()

    def on_rgbd_frame(depth, color):
        global pcl_converter

        depth_vis = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth_vis = cv2.equalizeHist(depth_vis)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)

        cv2.imshow("depth", depth_vis)
        cv2.imshow("color", color)
        # print(color.shape, depth.shape)
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        pcl_converter.rgbd_to_projection(depth, rgb, remove_noise=False)
        print(len(pcl_converter.pcl.points))
        pcl_converter.visualize_pcd()

    oak_d_pro = OAK(args.path)
    pcl_converter = PointCloudVisualizer(oak_d_pro.intrinsics, oak_d_pro.w, oak_d_pro.h)
    oak_d_pro.on_rgbd_frame = on_rgbd_frame

    while pcl_converter.is_running:
        oak_d_pro.update()

