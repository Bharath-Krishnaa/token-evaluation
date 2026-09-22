"""
Computes top1-hit@k, used by pruning_eval.py.

Different from occupancy@k (full set-overlap between the unmasked and
masked top-k sets): top1_hit@k only asks "does the UNMASKED model's top-1
token still show up in the MASKED model's top-k set?"

For one answer position:
    top1_hit@k = 1  if normal_top1(position) in top-k(masked_probs(position))
                = 0  otherwise
For one sample: top1_hit@k is the average of that 1/0 value over every
answer position -> one float per (sample, lambda, k). top1_hit_summary.csv
then averages those per-sample floats across samples, per lambda.

pruning_eval.py hands this checker the same base_embeddings /
final_mask_hard / normal_probs it already computed for occupancy@k, so
this does not repeat generation or mask training -- just one extra
forward pass to get masked_probs.
"""

import csv
import os
import statistics

import torch
import torch.nn.functional as F


def compute_top1_hit(model, base_embeddings, final_mask_hard, think_start, think_end,
                      pred_positions, normal_probs, max_k):
    # one extra forward pass with the frozen hard mask
    with torch.no_grad():
        embeddings = base_embeddings.clone()
        embeddings[:, think_start:think_end, :] = (
            base_embeddings[:, think_start:think_end, :] * final_mask_hard.unsqueeze(-1)
        )
        masked_logits = model(inputs_embeds=embeddings).logits
        masked_probs = F.softmax(masked_logits[:, pred_positions, :], dim=-1)

    num_answer_positions = normal_probs.shape[1]
    normal_top1 = torch.argmax(normal_probs[0], dim=-1)  # unmasked model's top-1 token per position

    top1_hit_per_k = {}
    for k in range(1, max_k + 1):
        masked_topk = torch.topk(masked_probs[0], k=k, dim=-1).indices
        hits = []
        for pos in range(num_answer_positions):
            masked_set = set(masked_topk[pos].tolist())
            hits.append(1.0 if normal_top1[pos].item() in masked_set else 0.0)
        top1_hit_per_k[k] = sum(hits) / len(hits)

    return top1_hit_per_k


class Top1HitChecker:
    def __init__(self, per_sample_csv, max_k):
        self.per_sample_csv = per_sample_csv
        self.max_k = max_k
        self.completed_pairs = self._load_completed_pairs()

        file_is_new = not os.path.exists(per_sample_csv)
        self.fieldnames = ["sample_idx", "lambda", "num_think_tokens", "kept_tokens",
                            "dropped_tokens", "skip_reason"] + \
                           [f"top1_hit@{k}" for k in range(1, max_k + 1)]
        self._file = open(per_sample_csv, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if file_is_new:
            self._writer.writeheader()
            self._file.flush()

    def _load_completed_pairs(self):
        completed = set()
        if not os.path.exists(self.per_sample_csv):
            return completed
        with open(self.per_sample_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    completed.add((int(row["sample_idx"]), float(row["lambda"])))
                except (KeyError, ValueError, TypeError):
                    continue
        return completed

    def is_done(self, sample_idx, lam):
        return (sample_idx, lam) in self.completed_pairs

    def _write_row(self, row):
        self._writer.writerow(row)
        self._file.flush()
        os.fsync(self._file.fileno())
        self.completed_pairs.add((row["sample_idx"], row["lambda"]))

    def record_skip(self, sample_idx, lam, num_think_tokens, skip_reason):
        if (sample_idx, lam) in self.completed_pairs:
            return
        row = {
            "sample_idx": sample_idx,
            "lambda": lam,
            "num_think_tokens": num_think_tokens,
            "kept_tokens": "",
            "dropped_tokens": "",
            "skip_reason": skip_reason,
            **{f"top1_hit@{k}": "" for k in range(1, self.max_k + 1)},
        }
        self._write_row(row)

    def record_result(self, model, base_embeddings, final_mask_hard, think_start, think_end,
                       pred_positions, normal_probs, sample_idx, lam, num_think_tokens, kept):
        if (sample_idx, lam) in self.completed_pairs:
            return

        top1_hit_per_k = compute_top1_hit(
            model, base_embeddings, final_mask_hard, think_start, think_end,
            pred_positions, normal_probs, self.max_k,
        )

        row = {
            "sample_idx": sample_idx,
            "lambda": lam,
            "num_think_tokens": num_think_tokens,
            "kept_tokens": kept,
            "dropped_tokens": num_think_tokens - kept,
            "skip_reason": "",
        }
        for k in range(1, self.max_k + 1):
            row[f"top1_hit@{k}"] = top1_hit_per_k[k]

        self._write_row(row)

        hit_str = ", ".join(f"k={k}:{top1_hit_per_k[k]:.3f}" for k in range(1, self.max_k + 1))
        print(f"    [top1_hit] {hit_str}")

    def close(self):
        self._file.close()

    def write_summary(self, summary_csv_path):
        # rebuilds summary_csv_path from scratch using whatever is currently
        # in per_sample_csv. mean_top1_hit@k averages the per-sample values
        # across samples for that lambda -- same two-level averaging as
        # occupancy@k.
        results = {}
        skipped_count = {}

        with open(self.per_sample_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lam = float(row["lambda"])
                except (KeyError, ValueError, TypeError):
                    continue
                results.setdefault(lam, {k: [] for k in range(1, self.max_k + 1)})
                skipped_count.setdefault(lam, 0)

                if row.get("skip_reason"):
                    skipped_count[lam] += 1
                    continue

                for k in range(1, self.max_k + 1):
                    val = row.get(f"top1_hit@{k}", "")
                    if val not in ("", None):
                        results[lam][k].append(float(val))

        summary_rows = []
        for lam in sorted(results.keys()):
            num_used = len(results[lam][1])
            num_skipped = skipped_count[lam]
            row = {"lambda": lam, "num_samples_used": num_used, "num_samples_skipped": num_skipped}
            for k in range(1, self.max_k + 1):
                values = results[lam][k]
                if len(values) == 0:
                    mean_v, std_v = float("nan"), float("nan")
                elif len(values) == 1:
                    mean_v, std_v = values[0], 0.0
                else:
                    mean_v = statistics.mean(values)
                    std_v = statistics.stdev(values)
                row[f"mean_top1_hit@{k}"] = mean_v
                row[f"std_top1_hit@{k}"] = std_v
            summary_rows.append(row)

        summary_fieldnames = ["lambda", "num_samples_used", "num_samples_skipped"] + \
            [f"mean_top1_hit@{k}" for k in range(1, self.max_k + 1)] + \
            [f"std_top1_hit@{k}" for k in range(1, self.max_k + 1)]
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_rows