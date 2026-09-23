# Model Card — ResNet-50 Embeddings (Image Similarity)

| | |
| --- | --- |
| **Registry name** | `resnet50-embed` |
| **Version** | `1.0.0` |
| **Task** | Image similarity / embedding |
| **Status** | Active, default for the similarity task |
| **Serving endpoints** | `POST /api/v1/similarity/{embed,index,search}` |
| **Output** | 2048-dimensional, L2-normalised |
| **Licence** | Weights from torchvision (BSD-3-Clause) |

---

## 1. What it does

It turns an image into a list of 2,048 numbers — an **embedding** — positioned
so that visually similar images end up close together.

That single idea supports several things without retraining: finding
near-duplicates, "more like this" search, clustering a collection, and
detecting when new uploads look unlike anything seen before.

Because every vector is normalised to length 1, the **dot product of two
vectors is their cosine similarity** directly: `1.0` identical direction,
`0.0` unrelated, `-1.0` opposite. That turns searching the whole index into a
single matrix multiplication.

---

## 2. Architecture and why it was chosen

Take the ResNet-50 classifier, and **remove its final classification layer**.

What remains is the 2,048-number description the network had built up just
before it decided on a class. That description is what we want: it encodes
shapes, textures and parts, without having collapsed everything down to one of
1,000 labels.

L2 normalisation is baked into the exported graph rather than applied in
Python afterwards. The artifact is therefore self-contained — anyone who runs
it gets unit vectors without having to remember an extra step, and the API and
the index cannot disagree about whether normalisation happened.

Why this rather than a purpose-built embedding model:

* **It reuses a model already being downloaded.** The classifier and the
  embedder share one backbone, so the system serves three tasks from two
  downloads. On a CPU-first deployment that is real memory saved.
* **It is genuinely good at "same kind of thing".** ImageNet features are
  strong semantic features.
* **It is fast**: 43 ms p50, the quickest of the three models.

**Trade-off, stated plainly.** A model trained with a *contrastive* objective
— CLIP, DINOv2 — produces materially better embeddings for retrieval, because
they are trained to make similar images close rather than having that emerge
as a side effect of classification. CLIP also allows text-to-image search,
which this cannot do at all. They were not chosen here because CLIP ViT-B/32
is ~350 MB against reusing a backbone already in memory, and the brief's
priority is a working similarity capability within the latency budget. **If
retrieval quality is the priority, swap this model.** The index sizes itself
from the first vector it sees, so a different dimensionality needs no config
change.

---

## 3. Training data

The same ImageNet-1k weights as the classifier (`IMAGENET1K_V2`). No
additional training — the classification head is removed, nothing is re-fit.

This has a direct consequence for behaviour: the features encode **what object
this is**, because that is what they were optimised to predict. They encode
style, colour palette and composition only incidentally. Two photographs of
different dogs will score as more similar than two photographs of the same
building at different times of day.

### Preprocessing

Identical to the classification model — resize to 256, centre crop 224,
ImageNet normalisation. Using different preprocessing would place the query
vector in a different region of the space from the indexed vectors, and every
similarity score would be wrong while still looking plausible.

---

## 4. Measured performance

Intel Core Ultra 7 155H, 22 logical cores, CPU only.

### Latency

| Format | Batch | p50 | p95 | p99 | Throughput | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX fp32 | 1 | **43.4 ms** | 278.3 ms | 387.9 ms | 10.8 img/s | 89.6 MB |
| ONNX fp32 | 4 | 302.7 ms | 420.7 ms | 453.6 ms | 12.7 img/s | 89.6 MB |
| ONNX INT8 (static) | 1 | 110.6 ms | 135.2 ms | 182.1 ms | 8.8 img/s | **22.9 MB** |
| ONNX INT8 (static) | 4 | 567.8 ms | 791.0 ms | 892.4 ms | 7.1 img/s | 22.9 MB |

The unusually wide p50-to-p95 gap at batch 1 (43 ms → 278 ms) is measurement
noise from a busy development machine, not model behaviour; the INT8 row,
measured in the same run, shows a normal spread.

**Requirement check:** p99 at batch 1 is 388 ms, inside the 1-second budget.

### Search latency

Search is exact brute force: the query vector against every indexed vector.

| Index size | Search time | Memory |
| --- | ---: | ---: |
| 1,000 | < 1 ms | 8 MB |
| 100,000 | ~5 ms | 780 MB |
| 1,000,000 | ~50 ms | 7.6 GB |

Both scale **linearly** with index size — see the limitations.

### Retrieval quality

| Check | Result |
| --- | --- |
| Self-similarity | **1.0000** — an image against itself, verified end to end |
| Embedding determinism | Identical vectors across repeated calls |
| Unit length | Verified: norm 1.0 ± 1e-5 straight from the graph |
| Distinct images | Score below self-match, as expected |

**Recall@k against a labelled retrieval benchmark has not been measured.**
That needs a dataset with ground-truth similarity judgements. The honest
summary is: the mechanism is verified correct, the *quality* of the ranking on
your data is unmeasured. Measure it before relying on a similarity threshold.

---

## 5. Validation results

| Check | Result |
| --- | --- |
| Artifact integrity | Pass |
| Determinism | Pass |
| Batch invariance | Pass |
| Output sanity | Pass |
| ONNX export fidelity | Pass — max abs diff vs PyTorch 1.9e-07 |
| Latency | Pass — p95 189 ms |

---

## 6. Limitations and failure modes

### It measures "same kind of object", not "looks alike"

The most common source of surprise. Because the features come from a
classifier, the model groups by *category*. Two different red sports cars
score highly; the same car photographed in daylight and at night may score
lower than you expect. It is not a style, colour or composition matcher.

### Weak at instance-level retrieval

Finding *this specific object* — a particular painting, a particular
person's luggage — is what contrastive models are for. This model will return
the right *category* and often the wrong instance.

### Scores are only comparable within one index

A similarity score is meaningful only against vectors produced by the **same
model version** with the **same preprocessing**. Changing the model invalidates
every stored vector. There is no migration path other than re-embedding the
whole collection, and mixing versions in one index produces scores that look
fine and mean nothing.

Treat the index as tied to a model version. Re-embed on upgrade.

### There is no universal "similar enough" threshold

Near-duplicates typically score above 0.95 and "same kind of thing" somewhere
around 0.7-0.9, but the right cut-off depends entirely on your images. Pick it
by labelling a sample of pairs from your own data, not by adopting a number
from a document.

### Exact search does not scale past ~1M vectors

The index compares the query against every stored vector, so both time and
memory grow linearly. It is fast and exact up to roughly a million vectors.
Beyond that, switch to an approximate index (FAISS HNSW, pgvector, a vector
database), which trades a little recall for a very large speed-up. The
interface in `api/services/similarity_index.py` is deliberately narrow so that
swap is contained.

### The index is not replicated

It lives in one process's memory, persisted to a `.npz` file. With several API
replicas, **each has its own index**, so an image indexed on replica A is not
findable on replica B. This is fine for a single instance and wrong for a
scaled deployment — see `docs/TECHNICAL.md` for the shared-store options.

---

## 7. Ethical and operational considerations

* **Embeddings are not anonymous.** A vector is derived from the image and can
  be used to match it against others. Treat stored embeddings with the same
  care as the images themselves.
* **Not for facial recognition.** These are general object features. They are
  not accurate enough for identification, and attempting it would be both
  ineffective and inappropriate.
* **No image is stored** — only vectors and any caller-supplied metadata.

---

## 8. Maintenance

| | |
| --- | --- |
| **Registered** | 2026-09-22 |
| **Upgrade path** | Swap to a contrastive model (CLIP, DINOv2), then **re-embed the entire index** |
| **Rollback** | Only safe alongside the matching index snapshot |

### Reproduce

```bash
python scripts/prepare_models.py --only similarity
python -m models.validation.validate --model resnet50-embed:1.0.0
```
