#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import subprocess
import tempfile

import lpips
import torch


def extract_frames(yuv_path: str,
                   frames_dir: str,
                   size: str,
                   in_pix_fmt: str,
                   fps: float,
                   out_pix_fmt: str = "rgb48be"):
    """
    Use ffmpeg to extract frames from a raw YUV video as PNG images.
    - yuv_path: path to the raw YUV file
    - frames_dir: directory where frames will be written
    - size: 'WxH' string, e.g. '1920x1080'
    - in_pix_fmt: input pixel format, e.g. 'yuv420p10le'
    - fps: frame rate (used to interpret the YUV sequence)
    - out_pix_fmt: output pixel format, e.g. 'rgb24' or 'rgb48be'
    """
    width, height = size.split("x")
    os.makedirs(frames_dir, exist_ok=True)

    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")

    cmd = [
        "ffmpeg",
        "-y",
        #"-loglevel", "quiet",
        "-f", "rawvideo",
        "-pix_fmt", in_pix_fmt,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", yuv_path,
        "-f", "image2",
        "-pix_fmt", out_pix_fmt,
        frame_pattern
    ]

    #print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def compute_lpips_for_dirs(ref_dir: str,
                           dist_dir: str,
                           net: str = "alex",
                           version: str = "0.1",
                           use_gpu: bool = False):
    """
    Compute LPIPS distance for all frame pairs in two directories.
    ref_dir / dist_dir: contain PNG files named frame_XXXXXX.png
    Returns: (mean_lpips, list_of_frame_scores)
    """
    loss_fn = lpips.LPIPS(net=net, version=version)
    if use_gpu:
        loss_fn.cuda()

    ref_frames = sorted(glob.glob(os.path.join(ref_dir, "frame_*.png")))
    dist_frames = sorted(glob.glob(os.path.join(dist_dir, "frame_*.png")))

    if not ref_frames:
        raise RuntimeError(f"No frames found in {ref_dir}")
    if not dist_frames:
        raise RuntimeError(f"No frames found in {dist_dir}")

    n = min(len(ref_frames), len(dist_frames))
    if len(ref_frames) != len(dist_frames):
        print(f"Warning: different number of frames "
              f"({len(ref_frames)} vs {len(dist_frames)}). "
              f"Using first {n} pairs.")

    scores = []

    print(f"Computing LPIPS on {n} frame pairs...")
    for i in range(n):
        im0 = lpips.im2tensor(lpips.load_image(ref_frames[i]))  # [-1, 1]
        im1 = lpips.im2tensor(lpips.load_image(dist_frames[i]))

        if use_gpu:
            im0 = im0.cuda()
            im1 = im1.cuda()

        with torch.no_grad():
            d = loss_fn(im0, im1)

        scores.append(float(d))

        # Optional: uncomment if you want per-frame debug
        # print(f"Frame {i+1:06d}: {scores[-1]:.6f}")

    mean_lpips = sum(scores) / len(scores)
    return mean_lpips, scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute LPIPS between two raw YUV videos using ffmpeg + LPIPS."
    )
    parser.add_argument("--ref", required=True,
                        help="Reference YUV file path.")
    parser.add_argument("--dist", required=True,
                        help="Distorted YUV file path.")
    parser.add_argument("--size", required=True,
                        help="Frame size as 'WxH', e.g. 1920x1080.")
    parser.add_argument("--in_pix_fmt", required=True,
                        help="Input pixel format, e.g. yuv420p10le.")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Frame rate of the YUV sequence (default: 25).")

    parser.add_argument("--out_pix_fmt", default="rgb48be",
                        help="Output pixel format for PNG extraction "
                             "(e.g. rgb24 or rgb48be). Default: rgb48be.")
    parser.add_argument("--net", default="alex",
                        choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone network (default: alex).")
    parser.add_argument("--version", default="0.1",
                        help="LPIPS version string (default: 0.1).")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for LPIPS if available.")

    parser.add_argument("--keep_frames", action="store_true",
                        help="Keep extracted PNG frames instead of deleting them.")
    parser.add_argument("--work_dir", default=None,
                        help="Optional directory to store intermediate frames. "
                             "If not set, a temporary directory is created.")

    args = parser.parse_args()

    # Working directory for frame extraction
    if args.work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="yuv_lpips_")
        auto_cleanup = True
    else:
        os.makedirs(args.work_dir, exist_ok=True)
        work_dir = args.work_dir
        auto_cleanup = False

    ref_frames_dir = os.path.join(work_dir, "ref_frames")
    dist_frames_dir = os.path.join(work_dir, "dist_frames")

    try:
        #print("=== Extracting reference frames ===")
        extract_frames(
            yuv_path=args.ref,
            frames_dir=ref_frames_dir,
            size=args.size,
            in_pix_fmt=args.in_pix_fmt,
            fps=args.fps,
            out_pix_fmt=args.out_pix_fmt,
        )

        #print("=== Extracting distorted frames ===")
        extract_frames(
            yuv_path=args.dist,
            frames_dir=dist_frames_dir,
            size=args.size,
            in_pix_fmt=args.in_pix_fmt,
            fps=args.fps,
            out_pix_fmt=args.out_pix_fmt,
        )

        #print("=== Computing LPIPS over all frames ===")
        mean_lpips, frame_scores = compute_lpips_for_dirs(
            ref_dir=ref_frames_dir,
            dist_dir=dist_frames_dir,
            net=args.net,
            version=args.version,
            use_gpu=args.use_gpu,
        )

        print(f"Number of frame pairs used: {len(frame_scores)}")
        print(f"Mean LPIPS: {(1-mean_lpips):.6f}")

    finally:
        if auto_cleanup and not args.keep_frames:
            print(f"Cleaning up temporary directory: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
