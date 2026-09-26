# Inference Benchmark Report

Generated: 2026-09-26T07:32:15.270354+00:00

## Environment

| Property | Value |
| --- | --- |
| method | 100 timed runs per case in 5 interleaved rounds, after 20 warmup runs |
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
| resnet50 | onnx | cpu | 1 | 75.11 | 149.83 | 211.70 | 72.43 | 13.8 | 97.4 |
| resnet50 | onnx | cpu | 4 | 244.29 | 387.22 | 445.39 | 66.08 | 15.1 | 97.4 |
| resnet50-embed | onnx | cpu | 1 | 69.14 | 124.93 | 357.34 | 79.08 | 12.6 | 89.6 |
| resnet50-embed | onnx | cpu | 4 | 246.23 | 370.93 | 421.29 | 65.44 | 15.3 | 89.6 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 1 | 28.09 | 99.34 | 132.28 | 39.83 | 25.1 | 22.9 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 4 | 96.63 | 126.62 | 173.09 | 23.79 | 42.0 | 22.9 |
| resnet50-tiny-imagenet | onnx | cpu | 1 | 66.80 | 132.76 | 277.08 | 72.99 | 13.7 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 4 | 271.37 | 449.44 | 553.27 | 72.82 | 13.7 | 91.2 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 1 | 39.54 | 99.56 | 149.00 | 49.23 | 20.3 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 4 | 103.99 | 215.60 | 273.89 | 29.46 | 33.9 | 23.3 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 1 | 201.46 | 347.58 | 488.18 | 222.83 | 4.5 | 23.1 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 4 | 820.77 | 1029.58 | 1370.34 | 209.86 | 4.8 | 23.1 |
| resnet50_int8_static | onnx_int8 | cpu | 1 | 36.95 | 97.11 | 141.64 | 44.83 | 22.3 | 24.9 |
| resnet50_int8_static | onnx_int8 | cpu | 4 | 102.87 | 142.97 | 195.48 | 26.98 | 37.1 | 24.9 |
| yolov8n | onnx | cpu | 1 | 113.05 | 226.19 | 307.50 | 136.06 | 7.3 | 12.1 |
| yolov8n | onnx | cpu | 4 | 396.86 | 670.27 | 829.31 | 105.65 | 9.5 | 12.1 |
| yolov8n_int8_static | onnx_int8 | cpu | 1 | 150.91 | 256.21 | 306.94 | 161.87 | 6.2 | 3.3 |
| yolov8n_int8_static | onnx_int8 | cpu | 4 | 523.50 | 975.16 | 1181.75 | 138.85 | 7.2 | 3.3 |

## Speed-up vs float32 ONNX (batch 1)

| Model | Runtime | p50 speed-up | Size reduction |
| --- | --- | ---: | ---: |
| resnet50-embed_int8_static | onnx_int8 | 2.46x | 3.91x |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | 1.69x | 3.91x |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | 0.33x | 3.94x |
| resnet50_int8_static | onnx_int8 | 2.03x | 3.92x |
| yolov8n_int8_static | onnx_int8 | 0.75x | 3.67x |

## How to read this

- **p50** is the typical request. **p95/p99** are the slow tail users complain about.
- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.
- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether
  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.
