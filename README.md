# token-evaluation

## Files

Qwen3-0.6B and MiniCPM5-1B use the same checker files, but each model has its own pruning and indicator files.

### Shared by both models
- `occupancy_checker.py` – occupancy@k
- `top1_hit_checker.py` – top1_hit@k
- `answer_consistency_checker.py` – checks if the answer with the oracle mask is the same as the answer without mask
- `total_loss_checker.py` – final KL, sparsity and total loss
- `train_indicator_cnn.py` – trains the CNN indicator
- `compare_oracle_indicator.py` – compares oracle mask and indicator mask consistency on the test samples

### Qwen3-0.6B
- `pruning_core.py` – model loading, generation and mask training
- `pruning_eval.py` – runs the pruning over GSM8K samples and calls the checkers
- `indicator_checker.py` – saves hidden states and oracle mask labels for the indicator
- `indicator_consistency_checker.py` – checks if the answer with the indicator mask is the same as the answer without mask

### MiniCPM5-1B
- `minicpm_pruning_core.py` – model loading, generation and mask training
- `minicpm_pruning_eval.py` – runs the pruning over GSM8K samples and calls the checkers
- `indicator_checker_minicpm.py` – saves hidden states and oracle mask labels for the indicator
- `indicator_consistency_checker_minicpm.py` – checks if the answer with the indicator mask is the same as the answer without mask

The pruning and indicator files are separate because MiniCPM5 handles `<think>` differently: its chat template already puts `<think>` in the prompt, so the model only writes the thinking text and `</think>`.

### Results
- `qwen_*.csv` and `minicpm5_*.csv` – per-sample results and summaries for occupancy, top1 hit, answer consistency and indicator consistency

- `oracle_vs_indicator_results/` – comparison of the oracle mask and the indicator mask for both models.
  The indicator is trained on the saved samples, which are split into train / val / test
  (by default 70% / 15% / 15%, fixed seed). The comparison is done only on the test samples,
  which the indicator never saw during training and only on test samples that have a result
  for both the oracle and the indicator.
  For each method we check whether its masked answer is the same as the answer without any mask
  (consistency) and how many thinking tokens it keeps (fraction_kept). The file also counts
  how many test samples are consistent for both methods such as only for the oracle, only for the
  indicator or for neither.




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
