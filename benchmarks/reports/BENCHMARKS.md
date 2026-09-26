# Inference Benchmark Report

Generated: 2026-09-26T03:13:13.763738+00:00

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
| resnet50 | onnx | cpu | 1 | 77.86 | 184.83 | 222.56 | 83.86 | 11.9 | 97.4 |
| resnet50 | onnx | cpu | 4 | 204.64 | 352.22 | 418.37 | 52.96 | 18.9 | 97.4 |
| resnet50-embed | onnx | cpu | 1 | 65.72 | 128.77 | 211.24 | 65.82 | 15.2 | 89.6 |
| resnet50-embed | onnx | cpu | 4 | 198.50 | 309.81 | 372.30 | 51.83 | 19.3 | 89.6 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 1 | 78.51 | 164.90 | 254.97 | 88.97 | 11.2 | 22.9 |
| resnet50-embed_int8_static | onnx_int8 | cpu | 4 | 360.69 | 507.34 | 592.28 | 87.61 | 11.4 | 22.9 |
| resnet50-tiny-imagenet | onnx | cpu | 1 | 66.62 | 138.98 | 176.58 | 65.87 | 15.2 | 91.2 |
| resnet50-tiny-imagenet | onnx | cpu | 4 | 207.49 | 311.68 | 333.06 | 53.51 | 18.7 | 91.2 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 1 | 76.76 | 156.48 | 268.94 | 83.24 | 12.0 | 23.3 |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | cpu | 4 | 340.68 | 560.15 | 687.66 | 88.73 | 11.3 | 23.3 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 1 | 174.90 | 311.40 | 360.06 | 184.89 | 5.4 | 23.1 |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | cpu | 4 | 689.35 | 981.87 | 1193.10 | 173.81 | 5.8 | 23.1 |
| resnet50_int8_static | onnx_int8 | cpu | 1 | 77.07 | 172.94 | 309.64 | 81.71 | 12.2 | 24.9 |
| resnet50_int8_static | onnx_int8 | cpu | 4 | 338.83 | 478.88 | 629.02 | 83.05 | 12.0 | 24.9 |
| yolov8n | onnx | cpu | 1 | 97.05 | 200.82 | 258.20 | 101.34 | 9.9 | 12.1 |
| yolov8n | onnx | cpu | 4 | 319.00 | 484.73 | 509.90 | 82.85 | 12.1 | 12.1 |
| yolov8n_int8_static | onnx_int8 | cpu | 1 | 177.38 | 278.11 | 342.77 | 183.35 | 5.5 | 3.3 |
| yolov8n_int8_static | onnx_int8 | cpu | 4 | 761.56 | 1104.26 | 1341.15 | 202.35 | 4.9 | 3.3 |

## Speed-up vs float32 ONNX (batch 1)

| Model | Runtime | p50 speed-up | Size reduction |
| --- | --- | ---: | ---: |
| resnet50-embed_int8_static | onnx_int8 | 0.84x | 3.91x |
| resnet50-tiny-imagenet_int8_static | onnx_int8 | 0.87x | 3.91x |
| resnet50-tiny-imagenet_int8_trt | onnx_int8 | 0.38x | 3.94x |
| resnet50_int8_static | onnx_int8 | 1.01x | 3.92x |
| yolov8n_int8_static | onnx_int8 | 0.55x | 3.67x |

## How to read this

- **p50** is the typical request. **p95/p99** are the slow tail users complain about.
- Latency below 1000 ms for batch 1 satisfies the challenge's sub-second requirement.
- INT8 usually wins on size and memory bandwidth; the speed-up depends on whether
  the CPU has INT8 acceleration (VNNI). Without it, INT8 can even be slower.
