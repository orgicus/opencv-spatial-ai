from depthai_sdk import Previews, FPSHandler
from depthai_sdk.managers import PipelineManager, PreviewManager, BlobManager, NNetManager
import depthai as dai
import cv2

# based on https://github.com/luxonis/depthai-experiments/blob/e93629ad5f7f642d160899120095edcc64895b02/gen2-yolo/car-detection/main.py
model_path  = "yolov7EscalatorsTiny/yolov7escalatorstiny_openvino_2021.4_6shave.blob"
config_path = "yolov7EscalatorsTiny/yolov7escalatorstiny.json"
w, h = 640, 640

# create pipeline manager and camera
pm = PipelineManager()
pm.createColorCam(previewSize=(w, h), xout=True)

# create yolo node
bm = BlobManager(blobPath=model_path)
nm = NNetManager(inputSize=(w, h), nnFamily="YOLO")
nm.readConfig(config_path)
nn = nm.createNN(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.color.name,
                 blobPath=bm.getBlob(shaves=6, openvinoVersion=pm.pipeline.getOpenVINOVersion()))
pm.addNn(nn)

# initialize pipeline
with dai.Device(pm.pipeline) as device:

    fpsHandler = FPSHandler()
    pv = PreviewManager(display=[Previews.color.name], fpsHandler=fpsHandler)

    pv.createQueues(device)
    nm.createQueues(device)

    nnData = []

    while True:
        # parse outputs
        pv.prepareFrames()
        inNn = nm.outputQueue.tryGet()

        if inNn is not None:
            # count FPS
            fpsHandler.tick("color")
            
            nnData = nm.decode(inNn)
            if len(nnData) > 0:
                print(dir(nnData[0]))
                nnData[0].xmin, nnData[0].ymin

        nm.draw(pv, nnData)
        pv.showFrames()

        if cv2.waitKey(1) == ord('q'):
            break

