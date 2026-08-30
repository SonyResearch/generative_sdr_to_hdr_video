import torch, os
import sys
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(_REPO_ROOT))
sys.path.append(str(_REPO_ROOT / "examples" / "wanvideo" / "model_training"))

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
from diffsynth.trainers.stuttgart_dataset import StuttgartDataset
from train_decoder import set_load_paths, load_models_from_paths
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
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
        ablation=None,
        use_vae_ea=False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        # Disable hub fetch — force local
        for cfg in model_configs:
            cfg.skip_download = True

        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        self.pipe.dit.require_vae_embedding = False  # Enable image VAE embeddings

        if ablation == "no_rope_add":
            from diffsynth.models.wan_video_dit import SelfAttention
            for module in self.pipe.dit.modules():
                if isinstance(module, SelfAttention):
                    module.no_rope_add = True
        

        if model_paths is not None:
            load_models_from_paths(model_paths, self.pipe)

        if use_vae_ea:
            from diffsynth.models.wan_video_vae_with_ea import VideoVAE38_WithEA
            old = self.pipe.vae.model
            ea_model = VideoVAE38_WithEA(
                dim=old.dim,
                z_dim=old.z_dim,
                dec_dim=old.decoder.dim,
                dim_mult=old.dim_mult,
                num_res_blocks=old.num_res_blocks,
                attn_scales=old.attn_scales,
                temperal_downsample=old.temperal_downsample,
            ).to(dtype=torch.bfloat16)
            ea_model.load_state_dict(old.state_dict(), strict=False)
            self.pipe.vae.model = ea_model

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.encoder_decoder_mode = encoder_decoder_mode
        self.use_vae_ea = use_vae_ea
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["bracket_video"],
            "exposures": data["exposures"],
            "height": data["bracket_video"][0].shape[0],
            "width": data["bracket_video"][0].shape[1],
            "num_frames": len(data["bracket_video"]),
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
            "input_type": data["input_type"],
            "test": False #train
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "condition_video":
                inputs_shared["condition_video"] = None
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


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
    checkpoints_use = []
    optimizers_use = []

    if args.model_paths is None:
        checkpoint_dir = Path(args.output_path) / "checkpoints"
        # find the latest checkpoint
        num_checkpoint_files = 0
        if checkpoint_dir.exists():  # use efa
            print("Looking for checkpoints in:", checkpoint_dir)
            checkpoint_files = list(checkpoint_dir.glob("epoch-*.safetensors"))
            num_checkpoint_files = len(checkpoint_files)
            print("Found {} checkpoint files.".format(num_checkpoint_files))
            if num_checkpoint_files > 0:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                latest_checkpoint = Path(latest_checkpoint)  # ensure it's a Path object
                checkpoints_use.append(str(latest_checkpoint))

                # also look for matching optimizer
                optimizer_path = checkpoint_dir / f"{latest_checkpoint.stem}.optimizer_scheduler.pt"
                if optimizer_path.exists():
                    optimizers_use.append(str(optimizer_path))
                else:
                    print("Warning: No matching optimizer checkpoint found.")

                print(f"Resuming from checkpoint: {latest_checkpoint}")
                epochs_done = int(latest_checkpoint.stem.split("-")[1]) + 1
                args.epochs_done = epochs_done
                got_checkpoint = True

        if not got_checkpoint:
            print("No checkpoints found, training from scratch.")
            args.epochs_done = 0

    if args.decoder_path is not None:
        print("Loaded decoder from:", args.decoder_path)
        checkpoints_use.append(args.decoder_path)

    args.model_paths = convert_strlist_to_jsonlist(checkpoints_use)
    args.optimizer_paths = optimizers_use

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
            crf_aug="random"
        ),
        mode="hdr_and_brackets",
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
        ),
        mode = "hdr_and_brackets",
        split = "val"
    )
    model = WanTrainingModule(
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
        ablation=getattr(args, "ablation", None),
    )
    model_logger = ModelLogger(
        Path(args.output_path) / "checkpoints",
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, val_dataset, model, model_logger, args=args)