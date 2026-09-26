# Inference Benchmark Report

Generated: 2026-09-25T16:21:54.582281+00:00

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
| resnet50 | onnx | cpu | 1 | 69.87 | 90.38 | 145.64 | 66.74 | 15.0 | 97.4 |
| resnet50 | onnx | cpu | 4 | 201.16 | 302.28 | 354.67 | 53.21 | 18.8 | 97.4 |
| resnet50-embed | onnx | cpu | 1 | 49.56 | 62.27 | 63.06 | 43.58 | 22.9 | 89.6 |
| resnet50-embed | onnx | cpu | 4 | 178.21 | 244.72 | 354.24 | 46.79 | 21.4 | 89.6 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 1 | 77.60 | 104.81 | 129.49 | 77.31 | 12.9 | 22.9 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 4 | 217.72 | 285.52 | 311.07 | 56.53 | 17.7 | 22.9 |
| resnet50-tiny-imagenet | onnx | cpu | 1 | 34.33 | 75.80 | 92.69 | 38.55 | 25.9 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 4 | 129.15 | 164.48 | 260.63 | 33.76 | 29.6 | 91.2 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 1 | 46.92 | 72.63 | 127.43 | 51.00 | 19.6 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 4 | 225.25 | 313.66 | 335.90 | 59.26 | 16.9 | 23.3 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 1 | 141.43 | 186.44 | 258.58 | 140.82 | 7.1 | 23.1 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 4 | 726.36 | 839.28 | 911.67 | 183.75 | 5.4 | 23.1 |
| resnet50_int8_static | onnx_int8 | cpu | 1 | 74.53 | 88.13 | 131.65 | 72.01 | 13.9 | 24.9 |
| resnet50_int8_static | onnx_int8 | cpu | 4 | 344.17 | 515.76 | 863.81 | 92.85 | 10.8 | 24.9 |
| yolov8n | onnx | cpu | 1 | 97.10 | 154.41 | 166.09 | 103.90 | 9.6 | 12.1 |
| yolov8n | onnx | cpu | 4 | 325.79 | 444.51 | 473.66 | 82.08 | 12.2 | 12.1 |
| yolov8n_int8_static | onnx_int8 | cpu | 1 | 223.56 | 309.43 | 337.58 | 228.55 | 4.4 | 3.4 |
| yolov8n_int8_static | onnx_int8 | cpu | 4 | 887.49 | 1133.01 | 1140.23 | 230.79 | 4.3 | 3.4 |

## Speed-up vs float32 ONNX (batch 1)

| Model | Runtime | p50 speed-up | Size reduction |
| --- | --- | ---: | ---: |
| resnet50-embed_int8_static | onnx_int8 | 0.64x | 3.91x |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | 0.73x | 3.91x |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | 0.24x | 3.94x |
| resnet50_int8_static | onnx_int8 | 0.94x | 3.92x |
| yolov8n_int8_static | onnx_int8 | 0.43x | 3.55x |

## How to read this

- **p50** is the typical request. **p95/p99** are the slow tail users complain about.
- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.
- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether
  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.
