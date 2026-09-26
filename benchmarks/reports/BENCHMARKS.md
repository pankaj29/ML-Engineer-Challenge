# Inference Benchmark Report

Generated: 2026-09-26T03:02:52.061422+00:00

## Environment

| Property | Value |
| --- | --- |
| platform | Windows-11-10.0.26200-SP0 |
| processor | Intel64 Family 6 Model 170 Stepping 4, GenuineIntel |
| cpu_count | 22 |
| python | 3.13.14 |
| torch | 2.9.0+cpu |
| onnxruntime | 1.20.1 |

## Results

Latency is wall-clock time for one forward pass. `per-image` divides by
batch size, which is the fair way to compare across batch sizes.

| Model | Runtime | Device | Batch | p50 (ms) | p95 (ms) | p99 (ms) | per-image (ms) | Throughput (img/s) | Size (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet50 | onnx | cpu | 1 | 61.31 | 82.53 | 88.10 | 51.26 | 19.5 | 97.4 |
| resnet50 | onnx | cpu | 4 | 180.93 | 277.34 | 373.24 | 49.96 | 20.0 | 97.4 |
| resnet50-embed | onnx | cpu | 1 | 52.00 | 88.16 | 124.06 | 53.85 | 18.6 | 89.6 |
| resnet50-embed | onnx | cpu | 4 | 182.01 | 256.22 | 291.97 | 47.85 | 20.9 | 89.6 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 1 | 58.01 | 90.98 | 130.76 | 60.52 | 16.5 | 22.9 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 4 | 235.85 | 347.24 | 413.55 | 61.96 | 16.1 | 22.9 |
| resnet50-tiny-imagenet | onnx | cpu | 1 | 36.10 | 39.43 | 58.52 | 37.36 | 26.8 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 4 | 136.51 | 213.92 | 256.44 | 38.52 | 26.0 | 91.2 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 1 | 43.32 | 70.89 | 79.92 | 47.93 | 20.9 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 4 | 220.47 | 394.62 | 459.15 | 59.89 | 16.7 | 23.3 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 1 | 106.07 | 177.15 | 201.29 | 113.25 | 8.8 | 23.1 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 4 | 419.67 | 596.52 | 651.49 | 109.71 | 9.1 | 23.1 |
| resnet50_int8_static | onnx_int8 | cpu | 1 | 77.33 | 116.08 | 125.16 | 74.81 | 13.4 | 24.9 |
| resnet50_int8_static | onnx_int8 | cpu | 4 | 197.59 | 317.94 | 356.66 | 53.80 | 18.6 | 24.9 |
| yolov8n | onnx | cpu | 1 | 77.03 | 132.07 | 146.73 | 82.09 | 12.2 | 12.1 |
| yolov8n | onnx | cpu | 4 | 281.86 | 394.44 | 484.97 | 74.04 | 13.5 | 12.1 |
| yolov8n_int8_static | onnx_int8 | cpu | 1 | 173.55 | 264.71 | 310.83 | 189.35 | 5.3 | 3.3 |
| yolov8n_int8_static | onnx_int8 | cpu | 4 | 704.44 | 1087.80 | 1334.52 | 192.03 | 5.2 | 3.3 |

## Speed-up vs float32 ONNX (batch 1)

| Model | Runtime | p50 speed-up | Size reduction |
| --- | --- | ---: | ---: |
| resnet50-embed_int8_static | onnx_int8 | 0.90x | 3.91x |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | 0.83x | 3.91x |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | 0.34x | 3.94x |
| resnet50_int8_static | onnx_int8 | 0.79x | 3.92x |
| yolov8n_int8_static | onnx_int8 | 0.44x | 3.67x |

## How to read this

- **p50** is the typical request. **p95/p99** are the slow tail users complain about.
- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.
- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether
  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.
