# Model Card: ResNet-50 embeddings (image similarity)

| | |
| --- | --- |
| Registry name | `resnet50-embed`, version `1.0.0`, default for similarity |
| Endpoints | `POST /api/v1/similarity/{embed,index,search}` and `/upload` variants |
| Output | 2,048 numbers, L2-normalised |
| Weights | torchvision `IMAGENET1K_V2` (BSD-3-Clause) |

## 1. What it does

Turns an image into 2,048 numbers placed so that similar images land close
together. That supports near-duplicate detection, "more like this" search and
clustering without training anything.

Every vector has length 1, so the dot product of two vectors is their cosine
similarity: 1.0 for the same direction, 0 for unrelated. Searching the whole
index is one matrix multiplication.

## 2. Why this model

It is the ImageNet ResNet-50 with its final classification layer removed. What
is left is the description the network builds just before it picks a class.
L2 normalisation is part of the exported graph, so anyone running the ONNX
file gets unit vectors without an extra step.

It shares weights with the classifier and is fast. A contrastively trained
model (CLIP, DINOv2) would give better retrieval and CLIP would allow
text-to-image search, but CLIP ViT-B/32 is about 350 MB on top of a backbone
already loaded, and the brief asks for a working similarity feature within a
latency budget. If retrieval quality matters most, swap it: the index sizes
itself from the first vector, so a different width needs no configuration.

Preprocessing is the classifier's: resize to 256, centre crop 224, ImageNet
normalisation. Query and indexed vectors must come from the same model and
preprocessing, or every score is wrong while still looking plausible.

## 3. Measured performance

### Latency

Intel Core Ultra 7 155H, CPU only, ONNX Runtime 1.20.1, 100 runs per case
interleaved with the other models (`benchmarks/reports/BENCHMARKS.md`).

| Runtime | Batch | p50 | p95 | p99 | Throughput | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX fp32 | 1 | 65.7 ms | 128.8 ms | 211.2 ms | 15.2 img/s | 89.6 MB |
| ONNX fp32 | 4 | 198.5 ms | 309.8 ms | 372.3 ms | 19.3 img/s | 89.6 MB |
| ONNX INT8 | 1 | 78.5 ms | 164.9 ms | 255.0 ms | 11.2 img/s | 22.9 MB |
| ONNX INT8 | 4 | 360.7 ms | 507.3 ms | 592.3 ms | 11.4 img/s | 22.9 MB |

### Search

Exact search over the in-process index, 50 queries per size
(`benchmarks/reports/similarity_search.json`):

⟦SEARCH_TABLE⟧

Time and memory both grow linearly with the number of vectors.

### Correctness

- The ONNX file matches PyTorch to a maximum absolute difference of
  5.96e-07 on the three sample photos (`benchmarks/reports/onnx_export.json`).
- INT8 vectors have a mean cosine similarity of 0.985 to the fp32
  vectors on 500 images (`benchmarks/reports/int8_fidelity.json`).
- Validation passes: deterministic, batch-invariant, no NaN or Inf, p95
  96.3 ms (`benchmarks/reports/validation.json`).

Retrieval quality (recall@k) is not measured; that needs a dataset with
ground-truth similarity judgements. The mechanism is verified, the ranking on
your data is not.

## 4. Limitations

- **It groups by kind of object, not by appearance.** Two different red cars
  score high; the same building by day and by night may not. It is not a
  style or colour matcher.
- **Weak at finding one specific object.** It returns the right category and
  often the wrong instance. That is what contrastive models are for.
- **Scores only mean something within one index.** Vectors from another model
  version or other preprocessing are incomparable. Re-embed everything on
  upgrade.
- **No universal threshold.** Pick a cut-off by labelling pairs from your own
  data.
- **Exact search is linear.** It stays fast to roughly a million vectors. Past
  that use an approximate index; pgvector offers HNSW.
- **The dev default is per process.** With `SIMILARITY_BACKEND=memory` each
  API replica has its own index and nothing survives a restart. Production and
  Kubernetes use pgvector, which is shared and persistent.
- **Embeddings are not anonymous.** They can be matched back to the image;
  store them with the same care. Not for facial recognition.

```bash
python scripts/prepare_models.py --only similarity
python -m models.validation.validate --model resnet50-embed:1.0.0
```
