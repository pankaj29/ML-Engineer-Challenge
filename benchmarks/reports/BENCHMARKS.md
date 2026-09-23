# Inference Benchmark Report

Generated: 2026-09-22T14:18:26.914899+00:00

## Environment

| Property | Value |
| --- | --- |
| platform | Windows-11-10.0.26200-SP0 |
| processor | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel |
| cpu_count | 22 |
| python | 3.13.14 |
| torch | 2.9.0+cpu |
| onnxruntime | 1.26.0 |

## Results

Latency is wall-clock time for one forward pass. `per-image` divides by
batch size, which is the fair way to compare across batch sizes.

| Model | Runtime | Device | Batch | p50 (ms) | p95 (ms) | p99 (ms) | per-image (ms) | Throughput (img/s) | Size (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet50 | onnx | cpu | 1 | 84.79 | 109.10 | 131.46 | 76.18 | 13.1 | 97.4 |
| resnet50 | onnx | cpu | 4 | 266.87 | 348.00 | 423.82 | 70.36 | 14.2 | 97.4 |
| resnet50-embed | onnx | cpu | 1 | 43.35 | 278.30 | 387.88 | 93.00 | 10.8 | 89.6 |
| resnet50-embed | onnx | cpu | 4 | 302.72 | 420.69 | 453.63 | 78.44 | 12.7 | 89.6 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 1 | 110.58 | 135.17 | 182.11 | 113.24 | 8.8 | 22.9 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 4 | 567.84 | 791.00 | 892.43 | 141.09 | 7.1 | 22.9 |
| resnet50_int8_static | onnx_int8 | cpu | 1 | 120.34 | 202.76 | 274.42 | 139.78 | 7.2 | 24.9 |
| resnet50_int8_static | onnx_int8 | cpu | 4 | 523.79 | 666.30 | 788.91 | 124.15 | 8.1 | 24.9 |
| yolov8n | onnx | cpu | 1 | 120.62 | 153.45 | 235.16 | 125.62 | 8.0 | 12.1 |
| yolov8n | onnx | cpu | 4 | 437.64 | 740.86 | 1097.28 | 116.55 | 8.6 | 12.1 |
| yolov8n_int8_static | onnx_int8 | cpu | 1 | 283.34 | 379.21 | 410.70 | 287.59 | 3.5 | 3.4 |
| yolov8n_int8_static | onnx_int8 | cpu | 4 | 1431.78 | 1841.59 | 2071.36 | 354.59 | 2.8 | 3.4 |

## Speed-up vs float32 ONNX (batch 1)

| Model | Runtime | p50 speed-up | Size reduction |
| --- | --- | ---: | ---: |

## How to read this

- **p50** is the typical request. **p95/p99** are the slow tail users complain about.
- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.
- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether
  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.
