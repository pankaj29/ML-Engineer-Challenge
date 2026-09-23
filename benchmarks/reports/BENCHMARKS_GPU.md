# Inference Benchmark Report

Generated: 2026-09-23T10:13:26.318915+00:00

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
| resnet50-tiny-imagenet | onnx | cpu | 1 | 11.50 | 11.64 | 11.65 | 11.50 | 86.9 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 8 | 64.90 | 65.39 | 65.70 | 8.11 | 123.3 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 32 | 252.30 | 258.95 | 260.50 | 7.93 | 126.2 | 91.2 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 1 | 17.09 | 17.51 | 17.58 | 17.10 | 58.5 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 8 | 85.66 | 87.26 | 88.02 | 10.71 | 93.4 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 32 | 318.56 | 325.08 | 325.16 | 9.22 | 108.5 | 23.3 |

## Speed-up vs float32 ONNX (batch 1)

| Model | Runtime | p50 speed-up | Size reduction |
| --- | --- | ---: | ---: |

## How to read this

- **p50** is the typical request. **p95/p99** are the slow tail users complain about.
- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.
- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether
  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.
