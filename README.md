# token-evaluation

## Data

The indicator dataset `indicator_data_v2` (1,763 `.pt` files, ~30 GB) is too large for GitHub and is hosted on the Hugging Face Hub:

https://huggingface.co/datasets/Bharath-Krishna-123/indicator_data_v2

To download it into `indicator_data_v2/` at the root of this repo:

```bash
pip install -U huggingface_hub
hf download Bharath-Krishna-123/indicator_data_v2 --repo-type dataset --local-dir indicator_data_v2
```

If the dataset is private, run `hf auth login` first with a token that has access to it.

Files are named `sample_<id>_lambda0.01.pt`. `*.pt` files are git-ignored, so the downloaded data will not be committed.
