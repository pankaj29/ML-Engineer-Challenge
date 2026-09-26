# Inference Benchmark Report

Generated: 2026-09-26T09:36:20.242522+00:00

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
| resnet50 | onnx | cpu | 1 | 30.09 | 107.57 | 143.40 | 45.63 | 21.9 | 97.4 |
| resnet50 | onnx | cpu | 4 | 105.05 | 260.05 | 307.01 | 33.59 | 29.8 | 97.4 |
| resnet50-embed | onnx | cpu | 1 | 28.44 | 79.51 | 84.08 | 38.76 | 25.8 | 89.6 |
| resnet50-embed | onnx | cpu | 4 | 118.22 | 287.39 | 322.03 | 39.77 | 25.1 | 89.6 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 1 | 14.07 | 81.38 | 165.44 | 27.66 | 36.2 | 22.9 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 4 | 60.20 | 103.51 | 160.63 | 17.12 | 58.4 | 22.9 |
| resnet50-embed_torch | torch | cpu | 1 | 80.37 | 280.62 | 332.12 | 114.57 | 8.7 | 89.7 |
| resnet50-embed_torch | torch | cpu | 4 | 201.00 | 660.07 | 763.08 | 71.32 | 14.0 | 89.7 |
| resnet50-tiny-imagenet | onnx | cpu | 1 | 30.01 | 109.78 | 137.23 | 50.40 | 19.8 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 4 | 123.73 | 286.98 | 335.27 | 40.94 | 24.4 | 91.2 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 1 | 14.74 | 73.45 | 177.05 | 26.78 | 37.3 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 4 | 48.02 | 99.23 | 111.60 | 14.34 | 69.7 | 23.3 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 1 | 96.56 | 229.72 | 275.32 | 118.29 | 8.5 | 23.1 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 4 | 345.95 | 822.55 | 856.60 | 117.74 | 8.5 | 23.1 |
| resnet50-tiny-imagenet_torch | torch | cpu | 1 | 81.12 | 274.56 | 330.92 | 116.03 | 8.6 | 91.2 |
| resnet50-tiny-imagenet_torch | torch | cpu | 4 | 198.30 | 631.79 | 724.87 | 69.72 | 14.3 | 91.2 |
| resnet50_int8_static | onnx_int8 | cpu | 1 | 13.92 | 65.59 | 146.09 | 24.42 | 41.0 | 24.9 |
| resnet50_int8_static | onnx_int8 | cpu | 4 | 43.62 | 132.87 | 161.15 | 15.19 | 65.8 | 24.9 |
| resnet50_torch | torch | cpu | 1 | 79.91 | 230.46 | 268.17 | 107.64 | 9.3 | 97.5 |
| resnet50_torch | torch | cpu | 4 | 201.33 | 573.08 | 625.41 | 66.50 | 15.0 | 97.5 |
| yolov8n | onnx | cpu | 1 | 46.93 | 141.73 | 186.84 | 66.45 | 15.0 | 12.1 |
| yolov8n | onnx | cpu | 4 | 183.59 | 396.14 | 434.59 | 54.47 | 18.4 | 12.1 |
| yolov8n_int8_static | onnx_int8 | cpu | 1 | 56.21 | 161.31 | 217.26 | 77.29 | 12.9 | 3.3 |
| yolov8n_int8_static | onnx_int8 | cpu | 4 | 227.20 | 455.22 | 484.56 | 66.47 | 15.0 | 3.3 |
| yolov8n_torch | torch | cpu | 1 | 90.49 | 253.50 | 285.65 | 116.40 | 8.6 | 12.0 |
| yolov8n_torch | torch | cpu | 4 | 242.76 | 768.06 | 800.59 | 83.34 | 12.0 | 12.0 |

## Speed-up vs float32 ONNX (batch 1)

| Model | Runtime | p50 speed-up | Size reduction |
| --- | --- | ---: | ---: |
| resnet50-embed_int8_static | onnx_int8 | 2.02x | 3.91x |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | 2.04x | 3.91x |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | 0.31x | 3.94x |
| resnet50_int8_static | onnx_int8 | 2.16x | 3.92x |
| yolov8n_int8_static | onnx_int8 | 0.83x | 3.67x |
| resnet50_torch | torch | 0.38x | 1.00x |
| resnet50-tiny-imagenet_torch | torch | 0.37x | 1.00x |
| resnet50-embed_torch | torch | 0.35x | 1.00x |
| yolov8n_torch | torch | 0.52x | 1.01x |

## How to read this

- **p50** is the typical request. **p95/p99** are the slow tail users complain about.
- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.
- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether
  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.
