# Learned Perceptual Video Patch Similarity (LPVPS) Metric

LPVPS is an extension of the original [Learned Perceptual Image Patch Similarity (LPIPS)](https://github.com/richzhang/PerceptualSimilarity/tree/master) metric, adapted to operate on video files instead of individual images.

LPIPS provides a learned, human-aligned measure of perceptual difference between two images. This project generalizes the idea to video by computing LPIPS on corresponding frame pairs and aggregating the results into a single similarity score.

## How LPVPS Works

Given two videos of equal length (reference and distorted):

1. The videos are decoded frame-by-frame (supporting any YUV/FFmpeg format, including 8-bit and 10-bit raw YUV).

2. LPIPS is computed for each pair of corresponding RGB frames.

3. Each LPIPS distance is transformed into a similarity score:
$$
s_i = 1 - \text{LPIPS}(f_i^\text{ref}, f_i^\text{dist})
$$

4. The final LPVPS score is the average over all N frames:
$$
\text{LPVPS} = \frac{1}{N} \sum_{i=1}^{N} s_i
$$

## Interpretation

LPVPS is designed to behave similarly to SSIM, but using the learned perceptual sensitivity of LPIPS.

* LPVPS = 1.0 → Videos are perceptually identical
* LPVPS ≈ 0.8 → Very high similarity
* LPVPS ≈ 0.5 → Moderate differences visible
* LPVPS → 0 → Strong perceptual degradation

This contrasts with the original LPIPS score, where higher means more different.
LPVPS flips the interpretation to be more intuitive for video-quality assessment.

## Why LPVPS?

* Works directly on raw YUV formats (via FFmpeg)
* Supports 10-bit and 8-bit content
* No temporary PNG files required (memory-efficient streaming mode)
* GPU-accelerated perceptual scoring via PyTorch
* Intuitive, similarity-oriented scale (compatible with SSIM/VMAF-style interpretation)

## Example Usage

```
python lpvps.py \
  --ref ref_video.yuv \
  --dist distorted_video.yuv \
  --size 1920x1080 \
  --in_pix_fmt yuv420p10le \
  --fps 60 \
  --use_gpu \
  --per_frame
```
Output:
```
Number of frame pairs: 600
Mean LPVS: 0.91234
```

## Notes
* LPVPS does not model temporal coherence (like flicker or judder) directly. It is strictly a per-frame perceptual similarity averaged over time.
* For a more complete video-quality assessment, LPVPS can be combined with temporal metrics or motion-aware models.
* This project remains interoperable with the original LPIPS model weights and networks (AlexNet, VGG, SqueezeNet).

## Dependencies/Setup

### Installation
* Install PyTorch 1.0+ and torchvision fom http://pytorch.org

```bash
pip install -r requirements.txt
```

* Download and install FFmpeg from https://www.ffmpeg.org/download.html

* Clone this repo:
```bash
git clone https://github.com/sergio-sanz-rodriguez/PerceptualSimilarityVideo
cd PerceptualSimilarityVideo
```