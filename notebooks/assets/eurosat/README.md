# EuroSAT tiles

Six 64 × 64 RGB tiles from the EuroSAT test split, one per class:
`annual_crop.png`, `forest.png`, `highway.png`, `residential.png`, `river.png`,
`sea_lake.png`. Each is the fourth test image of its class in the Hugging Face copy
[`tanganke/eurosat`](https://huggingface.co/datasets/tanganke/eurosat) (split `test`),
saved as PNG without other changes.

## Licence

**MIT**, per the EuroSAT repository <https://github.com/phelber/EuroSAT> ("The dataset is
licensed under the MIT license").

## Citation

- P. Helber, B. Bischke, A. Dengel, D. Borth. "EuroSAT: A Novel Dataset and Deep Learning
  Benchmark for Land Use and Land Cover Classification." *IEEE Journal of Selected Topics
  in Applied Earth Observations and Remote Sensing*, 2019.
- P. Helber, B. Bischke, A. Dengel. "Introducing EuroSAT: A Novel Dataset and Deep
  Learning Benchmark for Land Use and Land Cover Classification." *IGARSS 2018*.

Used in: `1_pytorch_intro/01b_tensors.ipynb` (the `(B, H, W, 3)` batch of images). The
EuroSAT vector cache `pttrn-io/eurosat-dinov2-small` is built from the same dataset.
