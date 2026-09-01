from networkx import center
import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import os
    
from pathlib import Path
import re
import cv2
import numpy as np
import os
from .stuttgart_dataset import ImageCropAndResize
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]




class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadPNGVideo(DataProcessingOperator):
    def __init__(self, num_frames=None, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, crf_aug=None):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.crf_aug = crf_aug
        self.cache = {}

        
    def __call__(self, folder: str):
        imgs = []
        
        files = sorted(os.listdir(folder))
        if self.num_frames is not None:
            files = files[:self.num_frames]

        for file in files:
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:,:,::-1].copy() #BGR to RGB
            img = self.frame_processor(img) #BGR to RGB
            imgs.append(img)
        
        exposures = np.array([0, 0, -1, 1]) #I always used these exposures
        return {"input_video": np.stack(imgs), "exposures": exposures}

class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        out_path = None,
        main_data_operator=lambda x: x,
        special_operator_map=None,
        type_filter=None,
    ):
        self.base_path = base_path
        self.out_path = out_path
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.cached_data = {}
        self.load_from_cache = False
        # Optional: only keep folders whose path (relative to base_path) starts
        # with one of these top-level subfolder names, e.g. ["auto", "over", "under"].
        self.type_filter = type_filter
        self.load_data_from_path()



    def load_data_from_path(self):
        self.data = []

        base = Path(self.base_path)

        folders_with_pngs = set()

        # Find every png file recursively
        for png_path in base.rglob("*.png"):
            folders_with_pngs.add(png_path.parent)

        if self.type_filter is not None:
            folders_with_pngs = {
                p for p in folders_with_pngs
                if p.relative_to(base).parts[0] in self.type_filter
            }

        # Store sorted folder list
        self.data = sorted([str(p) for p in folders_with_pngs])

    @staticmethod
    @staticmethod
    def default_video_operator(
        num_frames=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        crop_size_h=None, crop_size_w=None, 
        crf_aug=None,
    ):
        return LoadPNGVideo(
                num_frames=num_frames,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, crop_size_h=crop_size_h, crop_size_w=crop_size_w),
                crf_aug=crf_aug,
                )

            

    def __getitem__(self, data_id):
        folder_in = self.data[data_id]
        data = self.main_data_operator(folder_in)
        data["out_path"] = os.path.join(self.out_path, os.path.relpath(folder_in, self.base_path))
        data["prompt"] = ""
        return data

    def __len__(self):

        return len(self.data) 
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
