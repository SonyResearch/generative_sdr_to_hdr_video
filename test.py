import torch
from tqdm import tqdm
from accelerate import Accelerator
from diffsynth.trainers.utils import DiffusionTrainingModule
from accelerate.utils import DistributedDataParallelKwargs
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from pathlib import Path
from diffsynth.trainers.video_dataset import VideoDataset
from examples.wanvideo.model_training.train import WanTrainingModule, load_yaml_config, set_load_paths, load_models_from_paths
from utils import output_frames
from einops import rearrange

def launch_test_task(
    val_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    num_workers: int = 8,
    num_epochs: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        find_unused_parameters = args.find_unused_parameters
    
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        log_with="wandb",
    )
        
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers, persistent_workers=True)

    accelerator.init_trackers(
        project_name="hdrgen",
        init_kwargs={"wandb": {
            "name": "test",
        }}
    )
    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    with torch.no_grad():
        validate(val_dataloader, model, accelerator, val_dataset, args)


def validate(val_dataloader, model, accelerator, dataset, args):
    for step, data in enumerate(tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process)):
        with accelerator.accumulate(model):
            if dataset.load_from_cache:
                loss = model({}, inputs=data)
            else:
                condition_video = data["input_video"]
                unwrapped_model = accelerator.unwrap_model(model)
                import numpy as np
                outputs = unwrapped_model.pipe(
                    prompt=data["prompt"],
                    #negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    condition_video=condition_video,
                    height = args.height,
                    width = args.width,
                    num_inference_steps=50,
                    seed=1, tiled=False,
                    cfg_scale=1.0,
                    encoder_decoder_mode=unwrapped_model.encoder_decoder_mode,
                    exposures = data["exposures"],
                    generate_exposures = (-4, 0, 4),
                    use_vae_ea=unwrapped_model.use_vae_ea,
                )
                out_hdr_video = rearrange(outputs["hdr_video"], 'b c t h w -> b t c h w')
                output_frames(out_hdr_video[0], data["out_path"], mode="hdr", channel_order="NCHW")

                unwrapped_model.pipe.scheduler.set_timesteps(1000, training=True) #reset timesteps after inference
        print("Validation step completed, outputs generated.")

if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--dataset", type=str, default="stuttgart", choices=["stuttgart", "ubc"],
                         help="Evaluation dataset to test on.")
    parser.add_argument("--ae-types", type=str, default="auto,over,under",
                         help="Comma-separated exposure-mode subfolders to test, e.g. auto,over,under.")
    parser.add_argument("--method", type=str, default="ours",
                         help="Method name, used to name the output dir (default: ours).")
    parser.add_argument("--out-path", type=str, default=None,
                         help="Where to write predicted HDR frames (default: ./evaluations/<method>_<dataset>).")
    args = parser.parse_args()
    args = load_yaml_config(args, args.config)
    args = set_load_paths(args)

    ae_types = [t.strip() for t in args.ae_types.split(",") if t.strip()]
    out_path = args.out_path or f"./evaluations/{args.method}_{args.dataset}"

    val_dataset = VideoDataset(
        base_path=f"./evaluations/{args.dataset}",
        out_path=out_path,
        type_filter=ae_types,
        main_data_operator=VideoDataset.default_video_operator(
            num_frames=17,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
        )
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
        use_vae_ea=getattr(args, "use_vae_ea", False),
    )

    launch_test_task(
        val_dataset=val_dataset,
        model=model,
        num_workers=args.dataset_num_workers,
        num_epochs=args.num_epochs,
        find_unused_parameters=args.find_unused_parameters,
        args=args
    )