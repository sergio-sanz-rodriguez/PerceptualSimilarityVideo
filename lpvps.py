#!/usr/bin/env python3
import argparse
import subprocess
import numpy as np
import torch
import lpips
import warnings
import statistics

warnings.filterwarnings(
    "ignore",
    message=".*torch.load.*weights_only=False.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*The parameter 'pretrained' is deprecated since 0.13.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*Arguments other than a weight enum or `None` for 'weights'.*"
)
def make_ffmpeg_proc(yuv_path: str,
                     width: int,
                     height: int,
                     in_pix_fmt: str,
                     fps: float,
                     out_pix_fmt: str):
    """
    Launch ffmpeg to read raw YUV from file and write raw RGB frames to stdout.
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "error",  # keep it quiet
        "-f", "rawvideo",
        "-pix_fmt", in_pix_fmt,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", yuv_path,
        "-f", "rawvideo",
        "-pix_fmt", out_pix_fmt,
        "-"
    ]
    #print("Running:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return proc


def read_exact(stream, n_bytes: int) -> bytes:
    """
    Read exactly n_bytes from a stream, or fewer if EOF is reached.
    """
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = stream.read(n_bytes - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def frame_bytes(width: int, height: int, out_pix_fmt: str) -> int:
    """
    Compute number of bytes per RGB frame given output pixel format.
    Supports rgb24 (8-bit) and rgb48le (16-bit).
    """
    if out_pix_fmt == "rgb24":
        bytes_per_sample = 1
    elif out_pix_fmt == "rgb48le":
        bytes_per_sample = 2
    else:
        raise ValueError(f"Unsupported out_pix_fmt: {out_pix_fmt} (use rgb24 or rgb48le)")
    channels = 3
    return width * height * channels * bytes_per_sample


def buffer_to_tensor(buf: bytes,
                     width: int,
                     height: int,
                     out_pix_fmt: str,
                     device: torch.device) -> torch.Tensor:
    """
    Convert a raw RGB frame buffer to a torch Tensor of shape (1,3,H,W) in [-1,1].
    """
    if out_pix_fmt == "rgb24":
        dtype = np.uint8
        max_val = 255.0
    elif out_pix_fmt == "rgb48le":
        dtype = np.uint16
        max_val = 65535.0
    else:
        raise ValueError(f"Unsupported out_pix_fmt: {out_pix_fmt}")

    arr = np.frombuffer(buf, dtype=dtype)
    arr = arr.reshape((height, width, 3))  # H, W, C

    # Normalize to [0,1], then to [-1,1]
    arr = arr.astype(np.float32) / max_val
    arr = arr * 2.0 - 1.0

    # To tensor: (1,C,H,W)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def main():
    parser = argparse.ArgumentParser(
        description="Compute LPIPS between two raw YUV videos using ffmpeg streams (no PNGs on disk)."
    )
    parser.add_argument("--ref", required=True, help="Reference YUV file.")
    parser.add_argument("--dist", required=True, help="Distorted YUV file.")
    parser.add_argument("--size", required=True, help="Frame size 'WxH', e.g. 1920x1080.")
    parser.add_argument("--in_pix_fmt", required=True,
                        help="Input pixel format, e.g. yuv420p10le.")
    parser.add_argument("--fps", type=float, default=60.0,
                        help="Frame rate of the YUV sequence (default: 60).")

    parser.add_argument("--out_pix_fmt", default="rgb48le",
                        help="Output pixel format from ffmpeg (rgb24 or rgb48le). Default: rgb48le.")
    parser.add_argument("--net", default="alex",
                        choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone network (default: alex).")
    parser.add_argument("--version", default="0.1",
                        help="LPIPS version string (default: 0.1).")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for LPIPS if available.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Optional: limit number of frames to process.")
    parser.add_argument("--per_frame", action='store_true',
                        help="Print per-frame results.")

    args = parser.parse_args()

    width, height = map(int, args.size.split("x"))
    bytes_per_frame = frame_bytes(width, height, args.out_pix_fmt)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loss_fn = lpips.LPIPS(net=args.net, version=args.version).to(device)

    # Launch two ffmpeg processes (ref & dist)
    #print("=== Starting ffmpeg for reference video ===")
    proc_ref = make_ffmpeg_proc(
        yuv_path=args.ref,
        width=width,
        height=height,
        in_pix_fmt=args.in_pix_fmt,
        fps=args.fps,
        out_pix_fmt=args.out_pix_fmt,
    )

    #print("=== Starting ffmpeg for distorted video ===")
    proc_dist = make_ffmpeg_proc(
        yuv_path=args.dist,
        width=width,
        height=height,
        in_pix_fmt=args.in_pix_fmt,
        fps=args.fps,
        out_pix_fmt=args.out_pix_fmt,
    )

    frame_scores = []
    frame_idx = 0

    try:
        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                print(f"Reached max_frames={args.max_frames}, stopping.")
                break

            buf_ref = read_exact(proc_ref.stdout, bytes_per_frame)
            buf_dist = read_exact(proc_dist.stdout, bytes_per_frame)

            # Stop if either stream ended
            if len(buf_ref) < bytes_per_frame or len(buf_dist) < bytes_per_frame:
                if frame_idx == 0:
                    print("No frames read (check your size/pix_fmt/fps).")
                #else:
                #    print("End of stream reached.")
                break

            frame_idx += 1

            # Convert raw RGB buffers to tensors
            im0 = buffer_to_tensor(buf_ref, width, height, args.out_pix_fmt, device)
            im1 = buffer_to_tensor(buf_dist, width, height, args.out_pix_fmt, device)

            with torch.no_grad():
                d = loss_fn(im0, im1)

            score = 1.0 - float(d)
            frame_scores.append(score)

            if args.per_frame:
                #if frame_idx % 10 == 0:
                print(f"{frame_idx}: LPVPS={score:.6f}")

        if frame_scores:
            mean_lpvps = sum(frame_scores) / len(frame_scores)
            harm_lpvps = statistics.harmonic_mean(frame_scores)
            print("===========================")
            print(f"Number of frame pairs: {len(frame_scores)}")
            print(f"Mean LPVPS: {mean_lpvps:.6f} Arithmetic {harm_lpvps:.6f} Harmonic")            
        else:
            print("No valid frames processed.")

    finally:
        # Clean up ffmpeg processes
        for p, name in [(proc_ref, "ref"), (proc_dist, "dist")]:
            if p is not None and p.poll() is None:
                #print(f"Terminating ffmpeg process ({name})")
                p.terminate()
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    main()
