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

    for in_path in pbar:
        # Load person frame
        person_img = ensure_rgb(Image.open(in_path))

        # Optional global resize
        if args.size is not None:
            person_img = person_img.resize((args.size, args.size), Image.BICUBIC)
            ref_proc = ref_proc_base.resize((args.size, args.size), Image.BICUBIC)
        else:
            # match reference to current frame size
            ref_proc = ref_proc_base.resize(person_img.size, Image.BICUBIC)

        # Generate
        out_img = demo_mod.generate(person_img, ref_proc, args.obj_class, args.steps, args.guidance, args.seed)

        # Normalize output to PIL.Image
        if isinstance(out_img, (list, tuple)) and len(out_img) > 0:
            out_img = out_img[0]
        if not hasattr(out_img, "save"):
            arr = out_img
            if not isinstance(arr, np.ndarray):
                raise TypeError("Unexpected output type from generate(); cannot save.")
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype("uint8")
            out_img = Image.fromarray(arr)

        # Save per-frame
        if frames_out_dir is not None:
            out_frame_path = frames_out_dir / in_path.name
            if out_frame_path.exists() and not args.overwrite:
                pass  # skip
            else:
                out_img.save(out_frame_path)

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
