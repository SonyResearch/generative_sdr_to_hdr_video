from collections import deque
import numpy as np
import torch

DEPENDENCY = {+8: +4, -8: -4, -12: -8, 12: +8, 0: 0, -16: -12, +16: +12}  # others have no dependency
BASE_EXPOSURE = (0,-4,+4) #also will generate +4 and -4 as base exposures
FIRST_CHUNK = 5 # latent frames
NEXT_CHUNK = 3 # latent frames
TEMPORAL_STRIDE = 4
PREV_LATENT_FRAMES = 2 # frames of autoregressive context

def frames_to_latent_idx(f): return 0 if f == 0 else ((f - 1) // TEMPORAL_STRIDE) + 1
def latent_to_frame_idx(l):  return 0 if l == 0 else 1 + (l - 1) * TEMPORAL_STRIDE

PREV_FRAMES = latent_to_frame_idx(PREV_LATENT_FRAMES)


class ValScheduler:
    def __init__(self, condition_video, exposures, pipe, encoder_decoder_mode, tiled, tile_size, tile_stride, use_vae_ea=False):

        self.special_padding = False

        if self.special_padding:
            full_padded_video, self.pad_idx = [], []
            i, T = 0, condition_video.shape[0]

            while i < T:
                j = min(i + (17 if i == 0 else 8), T)
                full_padded_video += list(condition_video[i:j])
                full_padded_video += [condition_video[j-1]] * 4
                self.pad_idx += list(range(len(full_padded_video)-4, len(full_padded_video)))
                i = j

            full_padded_video = np.stack(full_padded_video)
            condition_video = full_padded_video
            
            #repeat 17th frame 4 times then 25th frame 5 times and 
        
        num_frames = len(condition_video)
        num_latents = frames_to_latent_idx(num_frames - 1) + 1

        # ===================== PAD VIDEO TO MATCH LATENT SCHEDULE =====================
        # ensure at least FIRST_CHUNK latents, then round up to FIRST_CHUNK + k*NEXT_CHUNK
        num_latents = max(num_latents, FIRST_CHUNK)
        num_latents = FIRST_CHUNK + ((num_latents - FIRST_CHUNK + NEXT_CHUNK - 1) // NEXT_CHUNK) * NEXT_CHUNK

        # latent length L corresponds to frame length T = 4*(L-1)+1
        target_frames = TEMPORAL_STRIDE * (num_latents - 1) + 1

        self.num_end_pad_frames = 0
        num_frames = condition_video.shape[0]

 
        # pad by repeating last frame
        if num_frames < target_frames:
            self.num_end_pad_frames = target_frames - num_frames
            pad = np.repeat(condition_video[-1:], self.num_end_pad_frames, axis=0)
            condition_video = np.concatenate([condition_video, pad], axis=0)

        num_frames = condition_video.shape[0]
        # ==============================================================================

        self.condition_video = condition_video
        self.exposures = exposures
        self.num_frames = num_frames
        self.num_latents = num_latents
        
        self.videos = None#create empty videos dict with correct shape once we know it from the first decoded latents 
        self.latents = None
        self.out_frames = None
        self.done_latents = {e: 0 for e in exposures}
        self.instruction = None

        self.pipe = pipe
        self.encoder_decoder_mode = encoder_decoder_mode
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.use_vae_ea = use_vae_ea

        self.latent_condition=True

    def exposure_complete(self, e):
        return self.done_latents[e] >= self.num_latents

    def dependency_ready(self, e):
        dep = DEPENDENCY.get(e, None)
        if dep is None:
            return True
        # You want conditioning latents for the same frame range, so dep must be at least as complete.
        return self.done_latents[dep] >= self.done_latents[e]

    def next_chunk_size_latents(self, e):
        if self.done_latents[e] == 0:
            return FIRST_CHUNK
        return NEXT_CHUNK

    def pick_exposure(self):
        # 1) finish base exposures first
        for e in BASE_EXPOSURE:
            if not (self.exposure_complete(e)) and self.dependency_ready(e):
                return e

        # 2) then generate +8 / -8 when their deps exist

        #grab +8 or -8 if in self.exposures
        larger_exps = [e for e in (+8, -8, +12, -12) if e in self.exposures]
        for e in larger_exps:
            if (not self.exposure_complete(e)) and self.dependency_ready(e):
                return e

        return None

    def generate_next(self):
        prev_frames_base, prev_frames_up, prev_frames_down = None, None, None
        e = self.pick_exposure()
        if e is None:
            return None  # nothing left to do

        l_start = self.done_latents[e]
        l_end = min(self.num_latents, l_start + self.next_chunk_size_latents(e))

        f_start = latent_to_frame_idx(l_start)
        f_end = latent_to_frame_idx(l_end)

        temporal_mode = "" if l_start == 0 else "_extend"

        # Conditioning for +8/-8 from +4/-4
        cond_exposure = DEPENDENCY.get(e, None)

        # # if cond_exposure is not None:
        # #     if self.latent_condition:
        # #         prev_frames_base = self.latents[cond_exposure][:, l_start:l_end]
        # #     else:
        # #         prev_frames_base = self.videos[cond_exposure][:, :, f_start:f_end]
        # #         prev_frames_base = self.pipe.vae.encode(self.pipe.preprocess_video(prev_frames_base[0].permute(1,2,3,0).cpu().float().numpy()*255), device=self.pipe.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        # if cond_exposure is None:
        
        if self.latent_condition:
            if cond_exposure == 0:
                video_latents =  self.pipe.vae.encode(self.pipe.preprocess_video(self.condition_video), device=self.pipe.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                video_latents = video_latents[:, :, max(0, l_start - PREV_LATENT_FRAMES): l_end]
            else:
                video_latents = self.latents[cond_exposure][:, max(0, l_start - PREV_LATENT_FRAMES): l_end]
        else:
            video_latents = None
        # Context for autoregressive continuation
        if temporal_mode == "_extend":

            if self.latent_condition:
                alpha = 0
                noise = torch.randn_like(self.latents[cond_exposure][:, max(0, l_start - PREV_LATENT_FRAMES):l_start]) * alpha
                prev_frames_base = self.latents[cond_exposure][:, max(0, l_start - PREV_LATENT_FRAMES):l_start] + noise
                prev_frames_down = self.latents[cond_exposure-4][:, max(0, l_start - PREV_LATENT_FRAMES):l_start] + noise
                prev_frames_up = self.latents[cond_exposure+4][:, max(0, l_start - PREV_LATENT_FRAMES):l_start] + noise

                # video_latents = video_latents[:, :, 0:5]
                # prev_frames_base = self.latents[cond_exposure][:, 0:2]
                # prev_frames_down = self.latents[cond_exposure-4][:, 0:2]
                # prev_frames_up = self.latents[cond_exposure+4][:, 0:2]
            else:
                if cond_exposure == 0:
                    prev_frames_base = self.videos[cond_exposure][:, :, max(0, f_start - PREV_FRAMES):f_start]
                else:
                    prev_frames_base = self.videos[cond_exposure][:, :, f_start- PREV_FRAMES:f_end]
                prev_frames_base = self.pipe.vae.encode(self.pipe.preprocess_video(prev_frames_base[0].permute(1,2,3,0).cpu().float().numpy()*255), device=self.pipe.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                prev_frames_down = self.videos[cond_exposure-4][:, :, max(0, f_start - PREV_FRAMES):f_start]
                prev_frames_up = self.videos[cond_exposure+4][:, :, max(0, f_start - PREV_FRAMES):f_start]
                prev_frames_down  = self.pipe.vae.encode(self.pipe.preprocess_video(prev_frames_down[0].permute(1,2,3,0).cpu().float().numpy()*255), device=self.pipe.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                prev_frames_up   = self.pipe.vae.encode(self.pipe.preprocess_video(prev_frames_up[0].permute(1,2,3,0).cpu().float().numpy()*255), device=self.pipe.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)


        crf_mode = "crf" if e==0 else "nocrf"
        input_type = crf_mode + temporal_mode

        self.instruction = {
            "video_segment": self.condition_video[f_start-PREV_FRAMES:f_end] if temporal_mode == "_extend" else self.condition_video[f_start:f_end],
            "video_latents": video_latents,
            "cond_exposure": cond_exposure,
            "generating_exposures": (e, e-4, e+4) if cond_exposure == 0 else (e,),
            "f_start": f_start,
            "f_end": f_end,
            "l_start": l_start,
            "l_end": l_end,
            "input_type": input_type,
            "prev_frames_base": prev_frames_base,
            "prev_frames_up": prev_frames_up, 
            "prev_frames_down": prev_frames_down,
        }
        return self.instruction

    def commit_result(self, new_latents):
        # new_latents must have length (end-start)
        num_latents_per_exposure = new_latents.shape[2] // 4
        crf_video = new_latents[:,:,0:num_latents_per_exposure]
        base_video = new_latents[:,:,1*num_latents_per_exposure:2*num_latents_per_exposure]
        low_video = new_latents[:,:,2*num_latents_per_exposure:3*num_latents_per_exposure]
        high_video = new_latents[:,:,3*num_latents_per_exposure:4*num_latents_per_exposure]

        self.create_latents(new_latents.shape[1:], new_latents.device, new_latents.dtype)
        self.create_videos(self.condition_video.shape, self.exposures, new_latents.device, new_latents.dtype)
    
        if self.instruction["cond_exposure"] in self.instruction["generating_exposures"]:
            self.latents[self.instruction["cond_exposure"]][:, self.instruction["l_start"]:self.instruction["l_end"]] = base_video[:,:,-(self.instruction["l_end"]-self.instruction["l_start"]):]
            self.done_latents[self.instruction["cond_exposure"]] = self.instruction["l_end"]
            self.videos[self.instruction["cond_exposure"]][:, :, self.instruction["f_start"]:self.instruction["f_end"]] = self.decode_latents(base_video)[:, :, - (self.instruction["f_end"]-self.instruction["f_start"]):]


        if self.instruction["cond_exposure"]-4 in self.instruction["generating_exposures"]:
            self.latents[self.instruction["cond_exposure"]-4][:, self.instruction["l_start"]:self.instruction["l_end"]] = low_video[:,:,-(self.instruction["l_end"]-self.instruction["l_start"]):]
            self.done_latents[self.instruction["cond_exposure"]-4] = self.instruction["l_end"]
            self.videos[self.instruction["cond_exposure"]-4][:, :, self.instruction["f_start"]:self.instruction["f_end"]] = self.decode_latents(low_video)[:, :, - (self.instruction["f_end"]-self.instruction["f_start"]):]


        if self.instruction["cond_exposure"]+4 in self.instruction["generating_exposures"]:
            self.latents[self.instruction["cond_exposure"]+4][:, self.instruction["l_start"]:self.instruction["l_end"]] = high_video[:,:,-(self.instruction["l_end"]-self.instruction["l_start"]):]
            self.done_latents[self.instruction["cond_exposure"]+4] = self.instruction["l_end"]
            self.videos[self.instruction["cond_exposure"]+4][:, :, self.instruction["f_start"]:self.instruction["f_end"]] = self.decode_latents(high_video)[:, :, - (self.instruction["f_end"]-self.instruction["f_start"]):]

    def create_latents(self, latents_shape, device, dtype):
        if self.latents is None:
            #normal_video = scheduler.latents[]
            #create empty latents dict with correct shape
            print(f"Creating latents with shape {latents_shape} for exposures {self.exposures}")
            self.latents = {e: torch.zeros((latents_shape[0], self.num_latents, latents_shape[2], latents_shape[3]), device=device, dtype=dtype) for e in self.exposures}

    def create_videos(self, video_shape, exposures, device, dtype):
        if self.videos is None:
            self.videos = {}
            for e in exposures:
                self.videos[e] = torch.zeros((1, video_shape[3], self.num_frames, video_shape[1], video_shape[2]), device=device, dtype=dtype)

    def decode_latents(self,latents):
        video = self.pipe.vae.decode(latents, device=latents.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride).to(dtype=torch.float32, device=latents.device)
        video = self.pipe.vae_output_to_video(video, mode="tensor")
        return video

    def _decode_latents_ea(self):
        sorted_exposures = sorted(self.exposures)
        E = len(sorted_exposures)
        all_latents = torch.stack([self.latents[e] for e in sorted_exposures], dim=0)  # (E, C, T', H', W')
        vae_device = next(self.pipe.vae.model.parameters()).device
        all_latents = all_latents.to(vae_device)
        with torch.no_grad():
            raw = self.pipe.vae.model.decode(all_latents, self.pipe.vae.scale, num_exposures=E, tiled=self.tiled)
        raw = raw.clamp_(-1, 1).to(dtype=torch.float32)
        return {e: self.pipe.vae_output_to_video(raw[i:i+1], mode="tensor") for i, e in enumerate(sorted_exposures)}

    def merge_and_output(self):
    #convert each video to float32
        if self.latent_condition:
            if self.use_vae_ea:
                self.videos = self._decode_latents_ea()
            else:
                self.videos = {e: self.decode_latents(self.latents[e].unsqueeze(0)) for e in self.exposures}
        else:
            self.videos = {e: self.videos[e].to(dtype=torch.float32) for e in self.exposures}

        videos_tensor = torch.stack([self.videos[e] for e in sorted(self.exposures)], dim=1)
        exposures = torch.tensor([e for e in sorted(self.exposures)], device=videos_tensor.device, dtype=videos_tensor.dtype)

        from utils import output_ldr_video
        for i, e in enumerate(sorted(self.exposures)):
            output_ldr_video(videos_tensor[0,i].permute(1,2,3,0), f"debug_video_{e}.mp4", fps=30)
        #remove padding frames if they exist
        if self.num_end_pad_frames > 0:
            videos_tensor = videos_tensor[:, :, :, :-self.num_end_pad_frames]


        if self.special_padding:
            #use the pad idx to remove the padded frames from the videos tensor before merging
            keep = torch.ones(videos_tensor.shape[3], dtype=torch.bool, device=videos_tensor.device)
            keep[self.pad_idx] = False
            videos_tensor = videos_tensor[:, :, :, keep]
                
        print("Merging videos with encoder-decoder mode:", self.encoder_decoder_mode)
        if self.encoder_decoder_mode == "latent":
            # Latent-space merger: merge raw latents before decoding, then decode once.
            # The decoder output for this mode is log-HDR, so it must be exponentiated.
            sorted_exposures = sorted(self.exposures)
            latents_tensor = torch.stack([self.latents[e].unsqueeze(0) for e in sorted_exposures], dim=1)
            merged_latents = self.pipe.merge_decoder(latents_tensor, exposures)
            log_hdr = self.pipe.vae.decode(merged_latents, device=merged_latents.device, tiled=self.tiled, tile_size=self.tile_size, tile_stride=self.tile_stride).to(dtype=torch.float32, device=merged_latents.device)
            hdr_video = torch.exp(log_hdr.clamp(min=-20, max=20))
            if self.num_end_pad_frames > 0:
                hdr_video = hdr_video[:, :, :-self.num_end_pad_frames]
        else:
            hdr_video = self.pipe.merge_decoder(videos_tensor, exposures, self.encoder_decoder_mode, mem_efficient=True)
        combined_video = torch.cat([self.videos[0], self.videos[-4], self.videos[4]], dim=2)
        torch.cuda.empty_cache()

        outputs = {
            "combined_video": combined_video,
            "hdr_video": hdr_video
        }

        return outputs
    
    

    def clean_up(self):
        del self.condition_video
        del self.latents 
        del self.instruction
        del self.videos
        torch.cuda.empty_cache()
