import imageio, os, torch, warnings, torchvision, argparse, json

from utils import generate_multi_exposure_video
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffsynth import save_video
import sys
from einops import rearrange
import wandb
from utils import vid_lpips


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import output_frames, output_ldr_video

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)

        def named_modules_recursive(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, child
                # recurse into the child
                yield from named_modules_recursive(child, prefix=full_name)
        
        def freeze_except_recursive(model, model_names):
            #freeze all first
            for name, child in named_modules_recursive(model):
                child.eval()
                child.requires_grad_(False)
            #then unfreeze the specified ones
            for name, model in named_modules_recursive(model):
                if name in model_names:
                    model.train()
                    model.requires_grad_(True)
                    print("Setting to training mode:", name)

        #special code for decoder
        if "vae" in trainable_models:
            freeze_except_recursive(pipe, trainable_models.split(","))
        else:
            # Freeze untrainable models
            pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone() #this creates a warning cause it wraps torch.barrier and this requires device ids
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            # keep only last 2 checkpoints
            files = sorted(
                [f for f in os.listdir(self.output_path) if f.startswith("epoch-")],
                key=lambda x: int(x.split("-")[1].split(".")[0])
            )
            for f in files[:-2]:
                os.remove(os.path.join(self.output_path, f))
                


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        epochs_done = args.epochs_done
        loss_type = args.loss_type
    

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        log_with="wandb",
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)

    accelerator.init_trackers(
        project_name="hdrgen",
        init_kwargs={"wandb": {
            "name": "decoderfinetune",
        }}
    )

    # 2. Initialize LPIPS
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    lpips_loss.eval()  # rec


    model, optimizer, dataloader, val_dataloader, scheduler, lpips_loss = accelerator.prepare(model, optimizer, dataloader, val_dataloader, scheduler, lpips_loss)
    

    accelerator.wait_for_everyone()
    for epoch_id in range(epochs_done, num_epochs):

        out_metrics = None 
        skip_val = False #debugging thing
        with torch.no_grad():
            saves = 0
            for step, data in enumerate(tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process)):
                if skip_val:
                    break
                #with accelerator.accumulate(model): I don't think we need this for validation
                if dataset.load_from_cache:
                    inputs, outputs = model({}, inputs=data)
                else:
                    inputs, outputs = model(data)


                from utils import get_psnr_fn, get_ssim_fn, get_lpips_fn, compute_all_metrics, average_metrics, flatten_dict

                
                outputs["dummy_string"] = "dummy" #add string to outputs to make it match data (this makes gathering same across both)
                data_gathered = accelerator.gather_for_metrics((data,), use_gather_object=False)
                outputs_gathered = accelerator.gather_for_metrics((outputs,), use_gather_object=False)
                

                #data, outputs = accelerator.gather_for_metrics((data, outputs), use_gather_object=True)
                if accelerator.is_main_process:
                    for i in range(len(data_gathered)):
                        device = outputs_gathered[0]["hdr_video"].device

                        if out_metrics is None:
                            psnr_fn = get_psnr_fn(device)
                            ssim_fn = get_ssim_fn(device)
                            lpips_fn = get_lpips_fn(device)

                            metrics = {
                                "low":  {"psnr": psnr_fn, "ssim": ssim_fn, "lpips": lpips_fn},
                                "mid":  {"psnr": psnr_fn, "ssim": ssim_fn, "lpips": lpips_fn},
                                "high": {"psnr": psnr_fn, "ssim": ssim_fn, "lpips": lpips_fn},
                                "hdr":  {"psnr": psnr_fn, "ssim": ssim_fn, "lpips": lpips_fn},
                            }
                            out_metrics = {
                            t: {m: torch.empty(0, device=device) for m in ["psnr", "ssim", "lpips"]}
                            for t in ["low", "mid", "high", "hdr"]
                        }
                            
                        data = data_gathered[i]
                        outputs = outputs_gathered[i]
                        out_dir = model_logger.output_path / "../" / "val_videos"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        

                        out_hdr_video = outputs["hdr_video"].to(device)
                        out_combined_video = outputs["combined_video"].to(device)
                        gt_hdr_video = torch.from_numpy(data["hdr_video"]).unsqueeze(0).to(device)
                        gt_combined_video = torch.from_numpy(data["bracket_video"]).unsqueeze(0).to(device)/255 #This scale feels funky here
                        gt_hdr_video = rearrange(gt_hdr_video, 'b t h w c -> b t c h w') 
                        gt_combined_video = rearrange(gt_combined_video, 'b t h w c -> b t c h w')
                        out_hdr_video = rearrange(out_hdr_video, 'b c t h w -> b t c h w')
                        out_combined_video = rearrange(out_combined_video, 'b c t h w -> b t c h w')


                        max_val = torch.max(gt_hdr_video)
                        normalized_out_hdr_video = out_hdr_video.to(torch.float32)/max_val
                        normalized_gt_hdr_video = gt_hdr_video.to(torch.float32)/max_val

                        from utils import generate_multi_exposure_video
                        arbitrary_scale_metrics = 16 #(if things  are normalized to 1 than this just lets me visualize correctlY)
                        out_multi_exposure_video = generate_multi_exposure_video(normalized_out_hdr_video*arbitrary_scale_metrics, exposures=(-4, 0, 4)) #using 16 here just for viz
                        gt_multi_exposure_video = generate_multi_exposure_video(normalized_gt_hdr_video*arbitrary_scale_metrics, exposures=(-4, 0, 4))

                        # ---- delete all outputs from last epoch on disk ----
                        if epoch_id >= 1:
                            import shutil, glob, os
                            prev_epoch = epoch_id
                            for f in glob.glob(str(out_dir / f"*epoch-{prev_epoch}_item-*.mp4")):
                                try: os.remove(f)
                                except FileNotFoundError: pass
                            for root in ["low_gt","mid_gt","high_gt","combined_gt","low_pred","mid_pred","high_pred","combined_pred","hdrnorm_gt","hdrnorm_pred","hdr_gt","hdr_pred"]:
                                for p in glob.glob(str(out_dir / root / f"*epoch-{prev_epoch}_item-*")):
                                    shutil.rmtree(p, ignore_errors=True)

                        # ---- exposure map ----
                        exposures = [("low", 0), ("mid", 1), ("high", 2)]
                        save_ldr = lambda tensor, rel: output_ldr_video(tensor, str(out_dir / rel), channel_order="NCHW")

                        for b in range(normalized_out_hdr_video.shape[0]):
                            # ----- metrics (now fully one-liner) -----
                            [compute_all_metrics(gt_multi_exposure_video[b, idx], out_multi_exposure_video[b, idx], name, metrics, out_metrics) for name, idx in exposures] + \
                            [compute_all_metrics(normalized_gt_hdr_video[b], normalized_out_hdr_video[b], "hdr", metrics, out_metrics)]

                            # ----- videos (fully one-liner too) -----
                            [save_ldr(gt_multi_exposure_video[b, idx], f"{name}_orig_epoch-{epoch_id+1}_item-{saves}.mp4") for name, idx in exposures] + \
                            [save_ldr(gt_combined_video[b], f"gt_combined_epoch-{epoch_id+1}_item-{saves}.mp4")] + \
                            [save_ldr(out_multi_exposure_video[b, idx], f"{name}_pred_epoch-{epoch_id+1}_item-{saves}.mp4") for name, idx in exposures] + \
                            [save_ldr(out_combined_video[b], f"out_combined_epoch-{epoch_id+1}_item-{saves}.mp4"),
                            save_ldr(normalized_gt_hdr_video[b],  f"hdrnorm_gt_epoch-{epoch_id+1}_item-{saves}.mp4"),
                            save_ldr(normalized_out_hdr_video[b], f"hdrnorm_pred_epoch-{epoch_id+1}_item-{saves}.mp4")]

                            # ----- frames (kept readable) -----
                            for name, idx in exposures:
                                output_frames(gt_multi_exposure_video[b, idx],  str(out_dir / f"{name}_gt" / f"gt_epoch-{epoch_id+1}_item-{saves}"),   mode="ldr", channel_order="NCHW")
                            output_frames(gt_combined_video[b],                 str(out_dir / "combined_gt"  / f"gt_epoch-{epoch_id+1}_item-{saves}"),  mode="ldr", channel_order="NCHW")
                            for name, idx in exposures:
                                output_frames(out_multi_exposure_video[b, idx], str(out_dir / f"{name}_pred"/ f"ours_epoch-{epoch_id+1}_item-{saves}"), mode="ldr", channel_order="NCHW")
                            output_frames(out_combined_video[b],                str(out_dir / "combined_pred"/ f"ours_epoch-{epoch_id+1}_item-{saves}"), mode="ldr", channel_order="NCHW")
                            output_frames(normalized_gt_hdr_video[b],           str(out_dir / "hdrnorm_gt"   / f"gt_epoch-{epoch_id+1}_item-{saves}"),  mode="ldr", channel_order="NCHW")
                            output_frames(normalized_out_hdr_video[b],          str(out_dir / "hdrnorm_pred" / f"ours_epoch-{epoch_id+1}_item-{saves}"),mode="ldr", channel_order="NCHW")
                            output_frames(normalized_gt_hdr_video[b],           str(out_dir / "hdr_gt"       / f"gt_epoch-{epoch_id+1}_item-{saves}"),  mode="hdr", channel_order="NCHW")
                            output_frames(normalized_out_hdr_video[b],          str(out_dir / "hdr_pred"     / f"ours_epoch-{epoch_id+1}_item-{saves}"),mode="hdr", channel_order="NCHW")

                            saves += 1
                        # end for b

                        #delete all outputs from last epoch on disk

            if accelerator.is_main_process and not skip_val:
                averaged_metrics = average_metrics(out_metrics)
                flattened_metrics = flatten_dict(averaged_metrics)
                accelerator.log(flattened_metrics)
                


        for data in tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}"):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                inputs, outputs = model({}, inputs=data, lpips_loss=lpips_loss) if dataset.load_from_cache else model(data)

                # Compute loss
                loss_dict = {}
                if loss_type == "hdr_l1":
                    eps = 1e-6
                    scale = inputs["hdr_video"].max()
                    hdr_loss = torch.nn.functional.l1_loss(torch.log(inputs["hdr_video"]/scale + eps), torch.log(outputs["hdr_video"]/scale + eps))
                    loss = hdr_loss
                    loss_dict["hdr_loss"] = hdr_loss
                elif loss_type == "hdr_multiexp":
                    eps = 1e-6
                    scale = 16.0
                    hdr_loss = torch.nn.functional.l1_loss(torch.log(outputs["hdr_video"]/scale + eps), torch.log(inputs["hdr_video"]/scale + eps))

                    normal_video_loss = torch.nn.functional.l1_loss(inputs["normal_video"], outputs["normal_video"])
                    short_video_loss = torch.nn.functional.l1_loss(inputs["short_video"], outputs["short_video"])
                    long_video_loss = torch.nn.functional.l1_loss(inputs["long_video"], outputs["long_video"])
                    loss = hdr_loss + normal_video_loss + short_video_loss + long_video_loss
                    loss_dict["hdr_loss"] = hdr_loss
                    loss_dict["normal_video_loss"] = normal_video_loss
                    loss_dict["short_video_loss"] = short_video_loss
                    loss_dict["long_video_loss"] = long_video_loss
                elif loss_type == "hdr_multilpips":
                    eps = 1e-6
                    scale = 16.0
                    hdr_loss = torch.nn.functional.l1_loss(torch.log(outputs["hdr_video"]/scale + eps), torch.log(inputs["hdr_video"]/scale + eps))
                    #multi exposure losses

                    normal_video_l1loss = torch.nn.functional.l1_loss(inputs["normal_video"], outputs["normal_video"])
                    short_video_l1loss = torch.nn.functional.l1_loss(inputs["short_video"], outputs["short_video"])
                    long_video_l1loss = torch.nn.functional.l1_loss(inputs["long_video"], outputs["long_video"])
                    normal_video_lpips = vid_lpips(outputs["normal_video"], inputs["normal_video"], lpips_loss)
                    short_video_lpips  = vid_lpips(outputs["short_video"],  inputs["short_video"],  lpips_loss)
                    long_video_lpips   = vid_lpips(outputs["long_video"],   inputs["long_video"],   lpips_loss)
                    lpips_alpha = 1
                    loss = hdr_loss + normal_video_l1loss + short_video_l1loss + long_video_l1loss + lpips_alpha*(normal_video_lpips + short_video_lpips + long_video_lpips)
                    loss_dict["hdr_loss"] = hdr_loss
                    loss_dict["normal_video_l1loss"] = normal_video_l1loss
                    loss_dict["short_video_l1loss"] = short_video_l1loss
                    loss_dict["long_video_l1loss"] = long_video_l1loss
                    loss_dict["normal_video_lpips"] = normal_video_lpips
                    loss_dict["short_video_lpips"] = short_video_lpips
                    loss_dict["long_video_lpips"] = long_video_lpips

                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

                # only log on the step that actually syncs grads
                if accelerator.sync_gradients:
                    # Reduce total loss
                    loss_mean = accelerator.reduce(loss.detach(), reduction="mean")

                    # Reduce each entry in loss_dict
                    reduced_loss_dict = {}
                    for k, v in loss_dict.items():
                        if torch.is_tensor(v):
                            reduced_v = accelerator.reduce(v.detach(), reduction="mean")
                            reduced_loss_dict[f"train/{k}"] = reduced_v.item()
                        else:
                            # if it's already a float or something
                            reduced_loss_dict[f"train/{k}"] = float(v)

                    # Add total loss explicitly too
                    reduced_loss_dict["train/loss"] = loss_mean.item()

                    # Log with wandb (only main process actually logs)
                    if accelerator.is_main_process:
                        accelerator.log(reduced_loss_dict)


        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
 
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    accelerator.init_trackers(
        project_name="hdrgen",
        init_kwargs={"wandb": {
            "name": "vae",
        }}
    )
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config", type=str, default="", required=True, help="Path to the config file.")
    return parser


