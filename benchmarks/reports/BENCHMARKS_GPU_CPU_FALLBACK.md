# Inference Benchmark Report

Generated: 2026-09-23T10:44:14.009491+00:00

## Environment

| Property | Value |
| --- | --- |
| platform | Linux-6.6.122+-x86_64-with-glibc2.39 |
| processor | x86_64 |
| cpu_count | 12 |
| python | 3.13.15 |
| torch | 2.11.0+cu128 |
| onnxruntime | 1.30.0 |
| gpu | NVIDIA A100-SXM4-40GB |
| cuda | 12.8 |

## Results

Latency is wall-clock time for one forward pass. `per-image` divides by
batch size, which is the fair way to compare across batch sizes.

| Model | Runtime | Device | Batch | p50 (ms) | p95 (ms) | p99 (ms) | per-image (ms) | Throughput (img/s) | Size (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet50-tiny-imagenet | onnx | cpu | 1 | 6.88 | 7.27 | 7.66 | 6.93 | 144.3 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 8 | 37.68 | 44.08 | 54.67 | 4.85 | 206.3 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 32 | 142.85 | 195.93 | 199.82 | 4.88 | 204.7 | 91.2 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 1 | 13.27 | 16.58 | 16.66 | 14.15 | 70.7 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 8 | 66.50 | 70.53 | 73.61 | 8.40 | 119.1 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 32 | 247.31 | 294.76 | 304.09 | 7.98 | 125.2 | 23.3 |

## Speed-up vs float32 ONNX (batch 1)

| Model | Runtime | p50 speed-up | Size reduction |
| --- | --- | ---: | ---: |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | 0.52x | 3.91x |

## How to read this

- **p50** is the typical request. **p95/p99** are the slow tail users complain about.
- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.
- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether
  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.
