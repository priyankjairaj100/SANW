# e-SNLI-VE release audit and reconstruction

The new experiment uses the current official e-SNLI-VE release accompanying
Kayser et al., ICCV 2021, from `maximek3/e-ViL`, immutable revision
`94f17f5c189573c9d1087f82eb27db37690fc810`.

The earlier `virginie-do/e-SNLI-VE` release was examined first. Its development
file contains 1,000 images; its test file contains 2,000 images, including all
1,000 development images. The two files share 11,896 pair IDs. Those overlapping
test rows contain 5,958 entailment and 5,938 contradiction labels, identical to
their development labels. The 1,000 test-only images have just 5,676 rows:
944 entailment, 1,195 contradiction, and 3,537 neutral. Excluding overlapping
images would therefore leave an incomplete test relation set. This release was
rejected before feature encoding, training, checkpoint selection or test scoring.
Its immutable revision, raw files, checksums and exact counts remain under
`raw/deprecated_2020_release/`.

The later official CSV release has 14,339 development rows across 1,000 images
and 14,740 test rows across 1,000 images, with zero image or pair overlap.
The builder also checks training-image separation and stops on any violation.

The earlier experiment's exact selected IDs were lost. This is a new
reconstruction. Python `random.Random(42)` samples from sorted official image
IDs in this sequence: 1,200 training images; 200 development images, assigning
the first 100 to calibration and the next 100 to validation; then 400 test
images. Each final split is stored in sorted image order. Every official
hypothesis annotation for every selected image is retained without filtering,
together with five original Flickr30k source captions in their stored order.
`selection.json` and `provenance.json` record the exact selected IDs and sources.

Images come from the public `nlphuji/flickr30k` mirror at immutable revision
`2b239befc81b6e3f035ce6bd52f5f4d60f5625f7`. Its image archive contains all 31,783
Flickr30k images. Its caption CSV, like the source Karpathy Flickr30k JSON,
contains only 31,014 images. This omission was detected before training and
no selected image was replaced. All source captions instead come from the
official `BryanPlummer/flickr30k_entities` repository at revision
`68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b`. Its `annotations.zip` contains all
31,783 sentence files, each with five captions. Entity annotation markers are
removed, preserving every original word, its case, and punctuation. Stored
tokens are joined by single spaces, following the official sentence parser.
The five file lines determine caption order.

The image archive
is accessed by verified HTTP byte ranges. Every selected image is checked
against the archive member's byte count and CRC32, decoded by Pillow for
structural verification, and recorded with a SHA256 checksum. No failed image
is silently skipped or replaced. Acquisition stops if any selected item fails.

Run from the repository root:

```sh
python scripts/prepare_visual_entailment.py --workers 16
```

The manifest uses image IDs `flickr:ID`, source-caption IDs `source:ID:0` through
`source:ID:4`, and hypothesis IDs containing the official split and pair ID.
The relation mapping is entailment to `supported`, contradiction to
`contradicted`, and neutral to `neutral`. Missing cross-image relations are left
unannotated. Hypothesis text is unchanged. Source-caption processing removes
only entity markup as described above.
