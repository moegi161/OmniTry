#!/usr/bin/env python3
"""
Infer eyeglass transfer on your own data using the baseline's Gradio demo code.

Inputs:
  - person frame (no glasses)
  - reference image of eyeglasses
  - optional binary mask for the eyeglasses in the reference image (white=keep, black=ignore)

This script imports `gradio_demo.py` (must be in the same repo) and calls its `generate(...)` function
directly, so you can run inference without launching the UI.

Usage:
  python infer_eyeglass_transfer.py 
      --person /path/to/person.jpg 
      --ref /path/to/eyeglasses_ref.jpg 
      --mask /path/to/eyeglasses_mask.png            # optional
      --class eyeglasses 
      --out /path/to/output.png 
      --steps 20 
      --guidance 30 
      --seed 1234

Notes:
  * If your baseline doesn't use a mask natively, we pre-apply the mask to the reference image so only the
    eyeglasses region is visible to the generator. If you omit --mask, the original ref image is used.
  * Run this script from the repository root (where `gradio_demo.py` can be imported and its relative
    paths like `configs/...` resolve). If needed, pass --demo-path to point to gradio_demo.py explicitly.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import torch
import math
import torchvision.transforms as T


# Optional natural sort
try:
    from natsort import natsorted as _natsorted
    def natsorted(x): return _natsorted(x)
except Exception:
    def natsorted(x): return sorted(x)


def load_module_from_path(py_path: Path, module_name: str = "gradio_demo"):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA"):
        return img.convert("RGB")
    return img.convert("RGB")


def prepare_ref_with_mask(ref_img: Image.Image, mask_img: Image.Image, feather: int = 3) -> Image.Image:
    """Apply a binary mask to the reference image so only eyeglasses remain.
    - mask white (>=128) = keep; black = remove
    - Optional feather (dilation + blur) to be tolerant to thin rims.
    """
    # Resize mask to reference size
    mask = mask_img.convert("L").resize(ref_img.size, Image.BILINEAR)

    # Slight dilation then blur for soft edges (helps thin frames)
    if feather > 0:
        # Dilation via MaxFilter, kernel size must be odd
        k = feather if feather % 2 == 1 else feather + 1
        mask = mask.filter(ImageFilter.MaxFilter(size=k))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))

    # Normalize to [0,255]
    mask = ImageOps.autocontrast(mask)

    # Composite reference onto a neutral background using mask as alpha
    bg = Image.new("RGB", ref_img.size, (0, 0, 0))  # black background
    cut = Image.composite(ref_img.convert("RGB"), bg, mask)
    return cut


def list_images(p: Path) -> List[Path]:
    if p.is_dir():
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        files = []
        for e in exts:
            files.extend(p.glob(e))
        return natsorted(files)
    elif p.is_file():
        return [p]
    else:
        raise FileNotFoundError(f"{p} is not a file or directory")


def cv2_writer(out_path: Path, fps: int, size: Tuple[int, int], codec: str = "mp4v"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path} with codec {codec}")
    return writer


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1]  # RGB->BGR


def run():
    ap = argparse.ArgumentParser(description="Sequence eyeglass transfer (frames+video) wrapper for gradio_demo.generate")
    ap.add_argument("--person", required=True, type=Path, help="Path to person frame (file) or directory of frames")
    ap.add_argument("--ref", required=True, type=Path, help="Path to reference eyeglasses image")
    ap.add_argument("--mask", type=Path, default=None, help="(Optional) Path to binary mask for eyeglasses in REF image")
    ap.add_argument("--class", dest="obj_class", default="glasses", help="Object class string for the baseline (default: glasses)")
    ap.add_argument("--frames-out", type=Path, default=None, help="Directory to save per-frame outputs (if omitted and --video-out is given, we'll derive from it)")
    ap.add_argument("--video-out", type=Path, default=None, help="Path to save MP4 video (optional)")
    ap.add_argument("--fps", type=int, default=30, help="FPS for output video")
    ap.add_argument("--codec", type=str, default="mp4v", help="FourCC codec for OpenCV VideoWriter (e.g., mp4v, avc1, H264)")
    ap.add_argument("--size", type=int, default=None, help="If set, resize frames and reference to size x size before generation")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed from a directory")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frame files")
    ap.add_argument("--steps", type=int, default=20, help="Diffusion steps (default: 20)")
    ap.add_argument("--guidance", type=float, default=30.0, help="Guidance scale (default: 30.0)")
    ap.add_argument("--seed", type=int, default=-1, help="Random seed; -1 for random (default: -1)")
    ap.add_argument("--demo-path", type=Path, default=None, help="(Optional) Explicit path to gradio_demo.py")
    args = ap.parse_args()

    # Resolve and import gradio_demo
    demo_py = args.demo_path
    if demo_py is None:
        guess = Path(__file__).resolve().parent / "gradio_demo.py"
        if not guess.exists():
            guess = Path.cwd() / "gradio_demo.py"
        if not guess.exists():
            raise FileNotFoundError("Could not find gradio_demo.py. Pass --demo-path /path/to/gradio_demo.py")
        demo_py = guess
    demo_mod = load_module_from_path(demo_py)

    if not hasattr(demo_mod, "generate"):
        raise AttributeError("gradio_demo.py does not expose a `generate(person_image, object_image, object_class, steps, guidance_scale, seed)` function.")

    # Load reference and prepare mask once (will resize per-frame as needed)
    ref_img = ensure_rgb(Image.open(args.ref))
    if args.mask is not None:
        mask_img = Image.open(args.mask)
        ref_proc_base = prepare_ref_with_mask(ref_img, mask_img, feather=3)
    else:
        ref_proc_base = ref_img

    # Collect frames
    frames_in = list_images(args.person)
    if len(frames_in) == 0:
        raise RuntimeError(f"No input frames found at {args.person}")

    if args.max_frames is not None:
        frames_in = frames_in[: args.max_frames]
        print(frames_in)

    is_dir_input = args.person.is_dir()

    # Derive default outputs
    frames_out_dir: Optional[Path] = args.frames_out
    if frames_out_dir is None and args.video_out is not None:
        frames_out_dir = args.video_out.with_suffix("")
    if frames_out_dir is None and is_dir_input:
        frames_out_dir = args.person.parent / f"{args.person.name}_out"

    # Prepare VideoWriter later when we know frame size
    writer = None
    video_out = args.video_out

    if frames_out_dir is not None:
        frames_out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(frames_in, desc="Processing", unit="frame")
    first_frame_size = None

        # ---- Batched inference over all frames ----
    import gradio_demo as demo_mod  # already imported at top, but safe if you keep it there

    if len(frames_in) == 0:
        raise ValueError(f"No input frames found under {args.person}")

    # Load all person frames
    person_imgs: List[Image.Image] = [ensure_rgb(Image.open(p)) for p in frames_in]

    # Optional global resize (to keep everything square and small enough)
    if args.size is not None:
        person_imgs = [
            img.resize((args.size, args.size), Image.BICUBIC) for img in person_imgs
        ]
        ref_img_resized = ref_proc_base.resize((args.size, args.size), Image.BICUBIC)
    else:
        # Use size of first frame as common size
        base_size = person_imgs[0].size  # (W, H)
        person_imgs = [
            img.resize(base_size, Image.BICUBIC) for img in person_imgs
        ]
        ref_img_resized = ref_proc_base.resize(base_size, Image.BICUBIC)

    # Seed like gradio_demo.generate
    from gradio_demo import seed_everything
    if args.seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed
    seed_everything(seed)

    # Resize model resolution as in gradio_demo.generate (based on first frame)
    max_area = 1024 * 1024
    oW, oH = person_imgs[0].width, person_imgs[0].height
    ratio = math.sqrt(max_area / (oW * oH))
    ratio = min(1, ratio)
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16

    transform_person = T.Compose([
        T.Resize((tH, tW)),
        T.ToTensor(),
    ])

    # Convert all person frames to a batched tensor
    person_tensors = [transform_person(img) for img in person_imgs]
    person_batch = torch.stack(person_tensors, dim=0)  # [N, 3, tH, tW]

    # Prepare reference (object) image: resize + center-pad once, then repeat
    object_image = ref_img_resized
    ratio_obj = min(tW / object_image.width, tH / object_image.height)
    transform_object = T.Compose([
        T.Resize(
            (int(object_image.height * ratio_obj), int(object_image.width * ratio_obj))
        ),
        T.ToTensor(),
    ])
    object_tensor = transform_object(object_image)  # [3, h', w']

    object_padded_single = torch.ones_like(person_tensors[0])
    new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_padded_single[:, min_y:min_y + new_h, min_x:min_x + new_w] = object_tensor

    """
    # Repeat object for each frame
    object_batch = object_padded_single.unsqueeze(0).repeat(person_batch.shape[0], 1, 1, 1)  # [N, 3, tH, tW]

    # Build img_cond by interleaving [person, object] as in gradio_demo.generate
    pairs = []
    for p_img, o_img in zip(person_batch, object_batch):
        pairs.append(p_img)
        pairs.append(o_img)
    img_cond = torch.stack(pairs, dim=0).to(
        dtype=demo_mod.weight_dtype,
        device=demo_mod.device,
    )  # [2N, 3, tH, tW]
    """
    
    # Build img_cond as (target_0, target_1, ..., target_{N-1}, reference)
    num_frames = person_batch.shape[0]

    # First all target frames
    imgs = [p_img for p_img in person_batch]  # length N

    # Then a single reference image at the end
    # (object_padded_single has the same size as person_tensors[0])
    imgs.append(object_padded_single)         # length N + 1

    img_cond = torch.stack(imgs, dim=0).to(
        dtype=demo_mod.weight_dtype,
        device=demo_mod.device,
    )  # [N+1, 3, tH, tW]
    

    # Zero mask for all samples
    mask = torch.zeros_like(img_cond, device=demo_mod.device)

    # Prompts: same object text for every (person, object) pair
    prompts = [demo_mod.args.object_map[args.obj_class]] * img_cond.shape[0]

    # Run the FluxFill pipeline once for the whole batch
    with torch.no_grad():
        result = demo_mod.pipeline(
            prompt=prompts,
            height=tH,
            width=tW,
            img_cond=img_cond,
            mask=mask,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=torch.Generator(demo_mod.device).manual_seed(seed),
        )
        all_images = result.images  # list of length 2N

    # Take only the "person" outputs (0, 2, 4, ...) as edited frames
    # out_images: List[Image.Image] = [all_images[2 * i] for i in range(len(person_imgs))]
    
    # Take only the target outputs (first N entries) as edited frames
    num_frames = len(person_imgs)
    out_images: List[Image.Image] = list(all_images[:num_frames])

    # ---- Save frames & (optional) video, as before ----
    first_frame_size = None
    for in_path, out_img in zip(frames_in, out_images):
        # Normalize output to PIL.Image (should already be PIL from pipeline)
        if isinstance(out_img, (list, tuple)) and len(out_img) > 0:
            out_img = out_img[0]
        if not hasattr(out_img, "save"):
            arr = out_img
            if not isinstance(arr, np.ndarray):
                raise TypeError("Unexpected output type from batched pipeline; cannot save.")
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype("uint8")
            out_img = Image.fromarray(arr)

        # Save per-frame
        if frames_out_dir is not None:
            out_frame_idx = frames_in.index(in_path) + 1
            out_frame_path = frames_out_dir / f"frame_{out_frame_idx:04d}.png"
            out_img.save(out_frame_path)

        # Write to video
        if video_out is not None:
            if writer is None:
                w, h = out_img.size
                first_frame_size = (w, h)
                writer = cv2_writer(video_out, fps=args.fps, size=(w, h), codec=args.codec)

            if out_img.size != first_frame_size:
                out_img = out_img.resize(first_frame_size, Image.BICUBIC)

            frame_bgr = pil_to_bgr(out_img)
            writer.write(frame_bgr)

        """
        # Init writer lazily
        if video_out is not None:
            if writer is None:
                w, h = out_img.size
                first_frame_size = (w, h)
                writer = cv2_writer(video_out, fps=args.fps, size=(w, h), codec=args.codec)

            # Ensure size consistency
            if out_img.size != first_frame_size:
                out_img = out_img.resize(first_frame_size, Image.BICUBIC)

            frame_bgr = pil_to_bgr(out_img)
            writer.write(frame_bgr)

    if writer is not None:
        writer.release()
    """

    print("Done.")
    if frames_out_dir is not None:
        print(f"Frames saved to: {frames_out_dir}")
    if video_out is not None:
        cmd = f"ffmpeg -y -r {args.fps} -i {frames_out_dir}/frame_%4d.png -vcodec libx264 -crf 11 -pix_fmt yuv420p {video_out}"
        os.system(cmd)
        print(f"Video saved to:  {video_out}")



if __name__ == "__main__":
    run()
