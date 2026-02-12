#!/usr/bin/env python3
"""
Per-frame inference wrapper for gradio_demo.

Instead of batch processing all frames together, this processes frames one-by-one.
Optionally uses the previous output as an anchor frame for temporal consistency.

Usage:
  python inference_per_frame.py \
      --person /path/to/frames/dir \
      --ref /path/to/ref.png \
      --mask /path/to/mask.png \
      --class glasses \
      --frames-out /path/to/output/dir \
      --steps 20 \
      --guidance 30 \
      --seed 1234 \
      --use-anchor-temporal  # Use previous frame as anchor
      --anchor-weight 2.0
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional

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
    """Apply a binary mask to the reference image so only eyeglasses remain."""
    mask = mask_img.convert("L").resize(ref_img.size, Image.BILINEAR)

    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        mask = mask.filter(ImageFilter.MaxFilter(size=k))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))

    mask = ImageOps.autocontrast(mask)
    bg = Image.new("RGB", ref_img.size, (0, 0, 0))
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


def run():
    ap = argparse.ArgumentParser(description="Per-frame eyeglass transfer with optional temporal anchoring")
    ap.add_argument("--person", required=True, type=Path, help="Path to person frame (file) or directory of frames")
    ap.add_argument("--ref", required=True, type=Path, help="Path to reference eyeglasses image")
    ap.add_argument("--mask", type=Path, default=None, help="(Optional) Path to binary mask for eyeglasses in REF image")
    ap.add_argument("--class", dest="obj_class", default="glasses", help="Object class string (default: glasses)")
    ap.add_argument("--frames-out", type=Path, default=None, help="Directory to save per-frame outputs")
    ap.add_argument("--size", type=int, default=None, help="If set, resize frames to size x size")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frame files")
    ap.add_argument("--steps", type=int, default=20, help="Diffusion steps (default: 20)")
    ap.add_argument("--guidance", type=float, default=30.0, help="Guidance scale (default: 30.0)")
    ap.add_argument("--seed", type=int, default=-1, help="Random seed; -1 for random")
    ap.add_argument("--demo-path", type=Path, default=None, help="(Optional) Explicit path to gradio_demo.py")
    
    # Anchoring options
    ap.add_argument("--anchor-frame", type=Path, default=None, help="(Optional) Fixed anchor frame image to use as reference")
    ap.add_argument("--anchor-weight", type=float, default=2.0, help="Weight for anchor donors (default: 2.0)")
    ap.add_argument("--start-frame-idx", type=int, default=0, help="Start from this frame index (0-indexed)")
    
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

    if not hasattr(demo_mod, "pipeline"):
        raise AttributeError("gradio_demo.py does not expose pipeline.")

    # Load reference and prepare mask once
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

    # Prepare output directory
    frames_out_dir: Optional[Path] = args.frames_out
    if frames_out_dir is not None:
        frames_out_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    from gradio_demo import seed_everything
    if args.seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = args.seed
    seed_everything(seed)

    # Get transform setup from first frame
    person_imgs: List[Image.Image] = [ensure_rgb(Image.open(p)) for p in frames_in]

    if args.size is not None:
        person_imgs = [img.resize((args.size, args.size), Image.BICUBIC) for img in person_imgs]
        ref_img_resized = ref_proc_base.resize((args.size, args.size), Image.BICUBIC)
    else:
        base_size = person_imgs[0].size
        person_imgs = [img.resize(base_size, Image.BICUBIC) for img in person_imgs]
        ref_img_resized = ref_proc_base.resize(base_size, Image.BICUBIC)

    # Compute target resolution
    oW, oH = person_imgs[0].width, person_imgs[0].height
    max_area = 1024 * 1024
    ratio = math.sqrt(max_area / (oW * oH))
    ratio = min(1, ratio)
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16

    transform_person = T.Compose([
        T.Resize((tH, tW)),
        T.ToTensor(),
    ])

    # Prepare reference (object) image once
    ratio_obj = min(tW / ref_img_resized.width, tH / ref_img_resized.height)
    transform_object = T.Compose([
        T.Resize((int(ref_img_resized.height * ratio_obj), int(ref_img_resized.width * ratio_obj))),
        T.ToTensor(),
    ])
    object_tensor = transform_object(ref_img_resized)

    object_padded = torch.ones(3, tH, tW)
    new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_padded[:, min_y:min_y + new_h, min_x:min_x + new_w] = object_tensor

    # Load anchor frame if provided
    anchor_tensor: Optional[torch.Tensor] = None
    if args.anchor_frame is not None:
        anchor_img = ensure_rgb(Image.open(args.anchor_frame))
        anchor_img = anchor_img.resize((oW, oH), Image.BICUBIC)
        if args.size is not None:
            anchor_img = anchor_img.resize((args.size, args.size), Image.BICUBIC)
        anchor_tensor = transform_person(anchor_img)
        print(f"Loaded anchor frame: {args.anchor_frame}")

    # Per-frame processing
    print(f"Processing {len(person_imgs)} frames...")

    for frame_idx, (in_path, person_img) in enumerate(zip(frames_in, person_imgs)):
        if frame_idx < args.start_frame_idx:
            print(f"Skipping frame {frame_idx} (before start index)")
            continue

        person_tensor = transform_person(person_img)

        # Build batch: [person, reference, (optional) anchor]
        imgs = [person_tensor, object_padded]
        
        # Use fixed anchor frame if provided
        anchor_used = False
        if anchor_tensor is not None:
            imgs.append(anchor_tensor)
            anchor_used = True

        img_cond = torch.stack(imgs, dim=0).to(
            dtype=demo_mod.weight_dtype,
            device=demo_mod.device,
        )

        mask = torch.zeros_like(img_cond, device=demo_mod.device)
        prompts = [demo_mod.args.object_map[args.obj_class]] * img_cond.shape[0]

        # Reference routing
        joint_attention_kwargs = {
            "ref_count": 1 if not anchor_used else 2,
            "anchor_weight": args.anchor_weight if anchor_used else 1.0,
        }

        # Run inference for this single frame
        print(f"\n[Frame {frame_idx:04d}] Processing... (anchor={anchor_used})")
        with torch.no_grad():
            result = demo_mod.pipeline(
                prompt=prompts,
                height=tH,
                width=tW,
                img_cond=img_cond,
                mask=mask,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=torch.Generator(demo_mod.device).manual_seed(seed + frame_idx),
                joint_attention_kwargs=joint_attention_kwargs,
            )
            all_images = result.images

        # Extract output (first image is the processed person)
        out_img = all_images[0]

        # Save frame
        if frames_out_dir is not None:
            out_frame_path = frames_out_dir / f"frame_{frame_idx:04d}.png"
            if isinstance(out_img, Image.Image):
                out_img.save(out_frame_path)
            else:
                arr = out_img
                if not isinstance(arr, np.ndarray):
                    raise TypeError("Unexpected output type")
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(out_frame_path)
            print(f"Saved: {out_frame_path}")

    print("\nDone.")
    if frames_out_dir is not None:
        print(f"Frames saved to: {frames_out_dir}")


if __name__ == "__main__":
    run()
