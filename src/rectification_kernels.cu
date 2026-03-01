/**
 * MIT License
 * Copyright (c) 2026 W.M. Nipun Dhananjaya
 *
 * @file rectification_kernels.cu
 * @brief CUDA kernel for bilinear image remapping (lens-distortion rectification).
 *
 * Usage:
 *   1. Precompute float32 remap maps (map_x, map_y) via cv::initUndistortRectifyMap
 *      on the CPU once at node startup, then upload them to device with cudaMemcpy.
 *   2. For each frame, call cuda_remap_bgra() to remap a packed BGRA image on the GPU.
 *
 * Both src and dst are expected to be packed (pitch = width * 4 bytes).
 * The VIC surface pitch-mismatch is handled by cudaMemcpy2D in the caller.
 */

#include <cuda_runtime.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// k_remap_bgra
//
// Bilinear pixel-remap kernel for packed BGRA images.
//
// For each destination pixel (x,y):
//   src_x = map_x[y*w + x]   (float, fractional source column)
//   src_y = map_y[y*w + x]   (float, fractional source row)
//
// Out-of-bounds source coordinates → black pixel (B=G=R=0, A=255).
// Bilinear interpolation over the 2×2 neighbourhood of (src_x, src_y).
//
// Grid  : 2-D, (ceil(w/16) × ceil(h/16)) blocks of 16×16 threads each.
// Shared : none — global-memory bandwidth dominates on Jetson iGPU (unified LPDDR5).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_remap_bgra(
  const uint8_t* __restrict__ src,   // packed BGRA input  (w*4 bytes/row)
  uint8_t*       __restrict__ dst,   // packed BGRA output (w*4 bytes/row)
  const float*   __restrict__ map_x, // float32 src-x map, row-major [h × w]
  const float*   __restrict__ map_y, // float32 src-y map, row-major [h × w]
  int w, int h)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  const int step = w * 4;  // packed row stride in bytes

  const float sx = map_x[y * w + x];
  const float sy = map_y[y * w + x];

  uint8_t* dp = dst + y * step + x * 4;

  // Out-of-bounds → black pixel
  if (sx < 0.0f || sy < 0.0f || sx >= static_cast<float>(w - 1) || sy >= static_cast<float>(h - 1)) {
    dp[0] = 0;   // B
    dp[1] = 0;   // G
    dp[2] = 0;   // R
    dp[3] = 255; // A
    return;
  }

  // Integer top-left corner of the 2×2 sample neighbourhood
  const int x0 = static_cast<int>(sx);
  const int y0 = static_cast<int>(sy);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;

  // Fractional offsets within the 2×2 cell
  const float fx = sx - static_cast<float>(x0);
  const float fy = sy - static_cast<float>(y0);

  // Bilinear weights
  const float w00 = (1.0f - fx) * (1.0f - fy);
  const float w10 = fx           * (1.0f - fy);
  const float w01 = (1.0f - fx) * fy;
  const float w11 = fx           * fy;

  // Pointers to the four neighbouring pixels
  const uint8_t* s00 = src + y0 * step + x0 * 4;
  const uint8_t* s10 = src + y0 * step + x1 * 4;
  const uint8_t* s01 = src + y1 * step + x0 * 4;
  const uint8_t* s11 = src + y1 * step + x1 * 4;

  // Bilinear blend for all 4 channels (B, G, R, A)
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float v = w00 * static_cast<float>(s00[c])
                  + w10 * static_cast<float>(s10[c])
                  + w01 * static_cast<float>(s01[c])
                  + w11 * static_cast<float>(s11[c]);
    dp[c] = static_cast<uint8_t>(__float2int_rn(v));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// cuda_remap_bgra   (extern "C" — callable from plain C++ translation units)
//
// Launches k_remap_bgra on the specified stream.
// Both d_src and d_dst must point to packed BGRA device buffers of w*h*4 bytes.
// d_map_x / d_map_y must point to float[w*h] device arrays (row-major).
//
// The function is asynchronous: the caller must synchronise the stream before
// reading d_dst.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void cuda_remap_bgra(
  const uint8_t* d_src,
  uint8_t*       d_dst,
  const float*   d_map_x,
  const float*   d_map_y,
  int w, int h,
  cudaStream_t stream)
{
  const dim3 block(16, 16);
  const dim3 grid(
    (static_cast<unsigned>(w) + 15u) / 16u,
    (static_cast<unsigned>(h) + 15u) / 16u);
  k_remap_bgra<<<grid, block, 0, stream>>>(d_src, d_dst, d_map_x, d_map_y, w, h);
}
