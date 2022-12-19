# opencv-spatial-ai
OpenCV Luxonis Spatial AI Contest Submission

### YOLOv7 Tiny Escalator

scripts:
- `yolo_escalators_live.py` : runs live camera with `yolov7escalatorstiny_openvino_2021.4_6shave.blob` model
- `yolo_escalators_replay.py`: WIP attempt using (fails on nn setup), but attempts to load [on site recording](https://drive.google.com/file/d/1OLsdA7FZCgwWtPbNKeAugy_kOHwhKS6f/view?usp=sharing) to be unzipped in the `recordings` folder (1.7GB)

live demo recording:

<video src="https://user-images.githubusercontent.com/189031/208370689-e978f884-a63d-410c-aebe-b98bb558244e.mp4" controls="controls" style="max-width: 730px;">
</video>

### Open3D GPU explorations

The scripts in this section are WIP and require an NVIDIA GPU and Open3D built with GPU support.

For windows users with Python 3.10 and CUDA 11.7 I've compiled a pip wheel from source accessible [here](https://drive.google.com/file/d/1ZN37I0XuR2cNenAarRNhD1GsgGh89JGr/view?usp=sharing)

- `replay_point_cloud.py`: run via `python replay_point_cliud.py -p recordings\3-184430102131341300` (after unzipping the above in the `recordings` folder) -> currently via 


https://user-images.githubusercontent.com/189031/208370689-e978f884-a63d-410c-aebe-b98bb558244e.mp4

