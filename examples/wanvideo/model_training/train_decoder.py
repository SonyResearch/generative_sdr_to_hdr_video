import gc
import torch, os
import sys
import yaml
from pathlib import Path
import wandb
sys.path.append(str(Path(__file__).resolve().parents[3]))

from diffsynth import load_state_dict, save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_PromptEmbedder
from diffsynth.trainers.utils_decoder import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
from diffsynth.trainers.stuttgart_dataset_decoder import StuttgartDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_models_from_paths(model_paths, pipe):
    full_state_dict = {}

    if model_paths is not None:
        import json
        model_paths = json.loads(model_paths)
        for path in model_paths:
            
            state_dict = load_state_dict(path)
            # Remove "pipe." prefix from keys (if present)
            state_dict = {
                (k[len("pipe."):] if k.startswith("pipe.") else k): v
                for k, v in state_dict.items()
            }
            full_state_dict.update(state_dict)
    #build a dict of all the first parts of the state dict keys
    all_keys_in_checkpoint = set()
    for key in full_state_dict.keys():
        first_part = key.split(".")[0]
        all_keys_in_checkpoint.add(first_part)


    print("Loading finetuned checkpoints for ", all_keys_in_checkpoint)
    missing, unexpected = pipe.load_state_dict(full_state_dict, strict=False)

    lora_path = model_paths[0]
    return lora_path

    #confirm the length of unexpected keys is 0
    #assert len(unexpected) == 0, f"Unexpected keys in state_dict: {unexpected}"


class WanDecoderTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        encoder_decoder_mode=None,
        loss_type="hdr_log",
        use_vae_ea=False,
        ea_num_heads=4,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        # Disable hub fetch — force local
        for cfg in model_configs:
            cfg.skip_download = True

        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        self.pipe.dit.require_vae_embedding = False  # Enable image VAE embeddings

        if model_paths is not None:
            load_models_from_paths(model_paths, self.pipe)

        # Swap the inner VAE model for an EA version before setting training mode
        if use_vae_ea:
            from diffsynth.models.wan_video_vae import VideoVAE38_
            from diffsynth.models.wan_video_vae_with_ea import VideoVAEWithEA_, VideoVAE38_WithEA
            old = self.pipe.vae.model
            ea_model = VideoVAE38_WithEA(
                dim=old.dim,
                z_dim=old.z_dim,
                dec_dim=old.decoder.dim,
                dim_mult=old.dim_mult,
                num_res_blocks=old.num_res_blocks,
                attn_scales=old.attn_scales,
                temperal_downsample=old.temperal_downsample,
                ea_num_heads=ea_num_heads,
            ).to(dtype=torch.bfloat16)
            ea_model.load_state_dict(old.state_dict(), strict=False)
            self.pipe.vae.model = ea_model

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Free DiT and language encoder — not used during decoder finetuning
        # Delete unused large modules to free memory.
        for _name in (
            "dit",
            "dit2",
            "text_encoder",
            "image_encoder",
            "motion_controller",
            "vace",
            "audio_encoder",
        ):
            if hasattr(self.pipe, _name):
                setattr(self.pipe, _name, None)

        gc.collect()
        torch.cuda.empty_cache()

        self.use_vae_ea = use_vae_ea

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.encoder_decoder_mode = encoder_decoder_mode
        self.loss_type = loss_type
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["input_video"],
            "height": data["input_video"][0].shape[0],
            "width": data["input_video"][0].shape[1],
            "num_frames": len(data["input_video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "encoder_decoder_mode": self.encoder_decoder_mode,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["input_video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["input_video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "bracket_video":
                inputs_shared["input_video"] = data["bracket_video"]
                inputs_shared["exposures"] = data["exposures"]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            if isinstance(unit, WanVideoUnit_PromptEmbedder):
                continue  # text_encoder freed; context unused in decoder finetuning
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def _decode_with_vae_ea(self, latents, exposures, tiled=False):
        """Decode all exposures together so EA can attend across them, then merge."""
        E = len(exposures)
        vae_device = next(self.pipe.vae.model.parameters()).device
        # latents: (B, E, C, T', H', W') — stack all E exposures into batch dim for EA
        all_latents = latents[0].to(vae_device)  # (E, C, T', H', W') for B=1
        # Call VideoVAEWithEA_.decode directly so num_exposures reaches the EA block
        with torch.no_grad():
            raw = self.pipe.vae.model.decode(all_latents, self.pipe.vae.scale, num_exposures=E, tiled=tiled)
        raw = raw.clamp_(-1, 1).to(dtype=torch.float32)
        videos = [
            self.pipe.vae_output_to_video(raw[i:i+1], mode="tensor")
            for i in range(E)
        ]
        hdr_video = self.pipe.merge_decoder(videos, exposures * 4, self.encoder_decoder_mode, mem_efficient=True)
        combined_video = torch.cat(videos, dim=2)
        return {"hdr_video": hdr_video, "combined_video": combined_video}

    def forward(self, data, inputs=None):
        with torch.no_grad():
            if inputs is None: inputs = self.forward_preprocess(data)
            hdr_video = self.pipe.preprocess_video( data["hdr_video"], min_value=0, max_value=1, in_max_value=1).to(self.pipe.device) #this keeps the scale the same
            bracket_video = self.pipe.preprocess_video( data["bracket_video"], min_value=0, max_value=1, in_max_value=255).to(self.pipe.device)
            encoded_latents = inputs["input_latents"]
        if self.use_vae_ea:
            outputs = self._decode_with_vae_ea(encoded_latents, data["exposures"], tiled=inputs["tiled"])
        else:
            outputs = self.pipe.decode_latents_hdr_merge(encoded_latents, self.encoder_decoder_mode, data["exposures"], tiled=inputs["tiled"], device=self.pipe.device)
        decoded_hdr_video = outputs["hdr_video"].to(torch.bfloat16)
        
        min_value = 0
        hdr_video_gt = torch.clamp(hdr_video, min=min_value)
        outputs["hdr_video"] = torch.clamp(decoded_hdr_video, min=min_value)

        num_frames = bracket_video.shape[2] // 4
        idx = 1
        inputs_data={"hdr_video": hdr_video_gt, "combined_video": bracket_video, "normal_video": bracket_video[:, :, (idx+0)*num_frames:(idx+1)*num_frames], "short_video": bracket_video[:, :, (idx+1)*num_frames:(idx+2)*num_frames], "long_video": bracket_video[:, :, (idx+2)*num_frames:(idx+3)*num_frames]}


        

        return inputs_data, outputs

def load_yaml_config(args, config_path):
    with open(config_path, "r") as f:
        config_args = yaml.safe_load(f) or {}

    for k, v in config_args.items():
        setattr(args, k, v)   # always set, no checks

    return args

#[\"./models/train/Wan2.2-TI2V-5B_full/epoch-0.safetensors\"]"
def convert_strlist_to_jsonlist(strlist):
    if strlist is None:
        return None
    import json
    # strlist is already a list → just dump it to JSON
    return json.dumps(strlist)
    


def set_load_paths(args):
    got_checkpoint = False
    if args.model_paths is None:
        checkpoint_dir = Path(args.output_path) / "checkpoints"
        #find the latest checkpoint
        num_checkpoint_files = 0
        if checkpoint_dir.exists(): #use efa
            print("Looking for checkpoints in:", checkpoint_dir)
            checkpoint_files = list(checkpoint_dir.glob("epoch-*.safetensors"))
            num_checkpoint_files = len(checkpoint_files)
            if num_checkpoint_files > 0:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                latest_checkpoint = Path(latest_checkpoint)  # ensure it's a Path object
                args.model_paths = convert_strlist_to_jsonlist([str(latest_checkpoint)])
                print(f"Resuming from checkpoint: {args.model_paths}")
                epochs_done = int(latest_checkpoint.stem.split("-")[1]) + 1
                args.epochs_done = epochs_done
                got_checkpoint = True
       
        if not got_checkpoint:
            default_model_id_with_origin_paths = ",Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
            args.model_id_with_origin_paths += default_model_id_with_origin_paths
            print("No checkpoints found, training from scratch.")
            args.epochs_done = 0
        return args
    else:
        return args

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    args = load_yaml_config(args, args.config)
    args = set_load_paths(args)

    dataset = StuttgartDataset(
        base_path=args.dataset_base_path,
        repeat=args.dataset_repeat,
        main_data_operator=StuttgartDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            crop_size_h = args.crop_size_h,
            crop_size_w = args.crop_size_w,
        ),
        mode = "hdr_and_brackets"
    )

    val_dataset = StuttgartDataset(
        base_path=args.dataset_base_path,
        repeat=args.dataset_repeat,
        main_data_operator=StuttgartDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            split="val"
        ),
        mode = "hdr_and_brackets",
        split = "val"
    )
    model = WanDecoderTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        encoder_decoder_mode=args.encode_decoder_mode,
        loss_type=args.loss_type,
        use_vae_ea=getattr(args, "use_vae_ea", False),
        ea_num_heads=getattr(args, "ea_num_heads", 4),
    )
    model_logger = ModelLogger(
        Path(args.output_path) / "checkpoints",
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, val_dataset, model, model_logger, args=args)
