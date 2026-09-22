"""
Computes occupancy@k, used by pruning_eval.py.

occupancy@k for one answer position:

overlap between the masked and unmasked model's top-k tokens, divided by k Averaged over all answer positions in a sample -> one number per (sample, lambda, k). 
1.0 means the masked model's top-k exactly matches the unmasked model's top-k at every position, 0.0 means no overlap.

"""

import csv
import os
import statistics

import torch
import torch.nn.functional as F


def compute_occupancy(model, base_embeddings, final_mask_hard, think_start, think_end,
                       pred_positions, normal_probs, max_k):
    # forward pass with the frozen hard mask applied
    with torch.no_grad():
        embeddings = base_embeddings.clone()
        embeddings[:, think_start:think_end, :] = (
            base_embeddings[:, think_start:think_end, :] * final_mask_hard.unsqueeze(-1)
        )
        masked_logits = model(inputs_embeds=embeddings).logits
        masked_probs = F.softmax(masked_logits[:, pred_positions, :], dim=-1)

    num_answer_positions = normal_probs.shape[1]
    occupancy_per_k = {}

    for k in range(1, max_k + 1):
        normal_topk = torch.topk(normal_probs[0], k=k, dim=-1).indices
        masked_topk = torch.topk(masked_probs[0], k=k, dim=-1).indices

        overlaps = []
        for pos in range(num_answer_positions):
            normal_set = set(normal_topk[pos].tolist())
            masked_set = set(masked_topk[pos].tolist())
            intersection = len(normal_set & masked_set)
            overlaps.append(intersection / k)

        occupancy_per_k[k] = sum(overlaps) / len(overlaps)

    return occupancy_per_k


class OccupancyChecker:
    def __init__(self, per_sample_csv, max_k):
        self.per_sample_csv = per_sample_csv
        self.max_k = max_k
        self.completed_pairs = self._load_completed_pairs()
        if self.completed_pairs:
            print(f"found {len(self.completed_pairs)} already-completed (sample_idx, lambda) row(s) "
                  f"in {per_sample_csv}, skipping those")

        file_is_new = not os.path.exists(per_sample_csv)
        self.fieldnames = ["sample_idx", "lambda", "num_think_tokens", "kept_tokens", "dropped_tokens",
                            "skip_reason"] + [f"occupancy@{k}" for k in range(1, max_k + 1)]
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
        os.fsync(self._file.fileno())  # write to disk right away
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
            **{f"occupancy@{k}": "" for k in range(1, self.max_k + 1)},
        }
        self._write_row(row)

    def record_result(self, model, base_embeddings, final_mask_hard, think_start, think_end,
                       pred_positions, normal_probs, sample_idx, lam, num_think_tokens, kept):
        if (sample_idx, lam) in self.completed_pairs:
            return None

        occupancy_per_k = compute_occupancy(
            model, base_embeddings, final_mask_hard, think_start, think_end,
            pred_positions, normal_probs, max_k=self.max_k,
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
            row[f"occupancy@{k}"] = occupancy_per_k[k]

        self._write_row(row)
        return occupancy_per_k

    def close(self):
        self._file.close()

    def write_summary(self, summary_csv_path):
        # rebuilds summary_csv from scratch using whatever is currently in
        # per_sample_csv, so it always reflects all rows collected so far
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
                    val = row.get(f"occupancy@{k}", "")
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
                row[f"mean_occupancy@{k}"] = mean_v
                row[f"std_occupancy@{k}"] = std_v
            summary_rows.append(row)

        summary_fieldnames = ["lambda", "num_samples_used", "num_samples_skipped"] + \
            [f"mean_occupancy@{k}" for k in range(1, self.max_k + 1)] + \
            [f"std_occupancy@{k}" for k in range(1, self.max_k + 1)]
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_rows