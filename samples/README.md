# Sample images

Three photographs so the examples in the documentation can be run as written,
without hunting for a picture first.

| File | Classification | Detection |
| --- | --- | --- |
| `dog.jpg` | Labrador retriever, 0.40 | dog, 0.82 |
| `street.jpg` | convertible, 0.23 | truck, 0.54 |
| `person.jpg` | stole, 0.06 | person, 0.65 |

`dog.jpg` is the one the docs use by default: it scores well on both the
classifier and the detector, so a single image exercises two endpoints.

`person.jpg` is a deliberately awkward case for the classifier. ImageNet-1k
has no "person" class, so a photo of someone gets forced into the nearest
clothing or object category with low confidence, while the detector finds the
person easily. It is a useful reminder that a confident-looking label is not
evidence the model understood the picture. See §6 of
[`resnet50-classification.md`](../models/cards/resnet50-classification.md).

## Where they came from

All three are from [Unsplash](https://unsplash.com), retrieved through
[Lorem Picsum](https://picsum.photos), and are used under the
[Unsplash licence](https://unsplash.com/license): free for commercial and
non-commercial use, no permission required. Attribution is not required but is
given here because it costs nothing.

| File | Photographer | Original |
| --- | --- | --- |
| `dog.jpg` | André Spieker | https://unsplash.com/photos/8wTPqxlnKM4 |
| `street.jpg` | Gabe Rodriguez | https://unsplash.com/photos/eLUegVAjN7s |
| `person.jpg` | Matthew Wiebe | https://unsplash.com/photos/U5rMrSI7Pn4 |

`SOURCES.json` holds the same mapping in machine-readable form, including the
Picsum id each was fetched by, so any of them can be re-downloaded:

```bash
curl -L -o samples/dog.jpg https://picsum.photos/id/237/640/480
```

Each is 640x480 and around 50 KB, so they are committed directly rather than
through Git LFS.
