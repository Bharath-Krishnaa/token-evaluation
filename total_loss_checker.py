"""
Computes the final training losses (KL, sparsity, total) for the frozen
mask, used by pruning_eval.py.

Reuses the same base_embeddings/final_mask_hard that pruning_core.train_mask
already produced -- one extra forward pass with the frozen mask, same KL +
sparsity formula train_mask uses during training, just evaluated once at the
end instead of every training step.

This file owns result.csv / result_summary.csv (whatever paths
--total_loss_per_sample_csv / --total_loss_summary_csv point at). Existing
rows are read on startup, so a run picks up where it left off, same as the
other checkers.
"""

import csv
import os
import statistics

import torch
import torch.nn.functional as F


def compute_final_losses(model, base_embeddings, final_mask_hard, think_start, think_end,
                          pred_positions, normal_probs, lambda_sparsity):
    with torch.no_grad():
        embeddings = base_embeddings.clone()
        embeddings[:, think_start:think_end, :] = (
            base_embeddings[:, think_start:think_end, :] * final_mask_hard.unsqueeze(-1)
        )
        masked_logits = model(inputs_embeds=embeddings).logits
        masked_log_probs = F.log_softmax(masked_logits[:, pred_positions, :], dim=-1)

        kl_per_token = F.kl_div(masked_log_probs, normal_probs, reduction="none").sum(dim=-1)
        kl_loss = kl_per_token.mean().item()
        sparsity_loss = final_mask_hard.mean().item()
        total_loss = kl_loss + lambda_sparsity * sparsity_loss

    return kl_loss, sparsity_loss, total_loss


class TotalLossChecker:
    def __init__(self, per_sample_csv):
        self.per_sample_csv = per_sample_csv
        self.completed_pairs = self._load_completed_pairs()
        if self.completed_pairs:
            print(f"found {len(self.completed_pairs)} already-completed (sample_idx, lambda) row(s) "
                  f"in {per_sample_csv}, skipping those")

        file_is_new = not os.path.exists(per_sample_csv)
        self.fieldnames = [
            "sample_idx", "status", "num_prompt_tokens", "num_think_tokens", "num_answer_tokens",
            "seq_len", "kept_tokens", "dropped_tokens", "kept_fraction",
            "final_kl_loss", "final_sparsity_loss", "final_total_loss", "lambda_sparsity",
        ]
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
                    completed.add((int(row["sample_idx"]), float(row["lambda_sparsity"])))
                except (KeyError, ValueError, TypeError):
                    continue
        return completed

    def is_done(self, sample_idx, lam):
        return (sample_idx, lam) in self.completed_pairs

    def _write_row(self, row):
        self._writer.writerow(row)
        self._file.flush()
        os.fsync(self._file.fileno())  # write to disk right away
        self.completed_pairs.add((row["sample_idx"], row["lambda_sparsity"]))

    def record_skip(self, sample_idx, lam, num_think_tokens, skip_reason):
        if (sample_idx, lam) in self.completed_pairs:
            return
        row = {
            "sample_idx": sample_idx,
            "status": skip_reason,
            "num_prompt_tokens": "",
            "num_think_tokens": num_think_tokens,
            "num_answer_tokens": "",
            "seq_len": "",
            "kept_tokens": "",
            "dropped_tokens": "",
            "kept_fraction": "",
            "final_kl_loss": "",
            "final_sparsity_loss": "",
            "final_total_loss": "",
            "lambda_sparsity": lam,
        }
        self._write_row(row)

    def record_result(self, model, base_embeddings, final_mask_hard, think_start, think_end,
                       pred_positions, normal_probs, sample_idx, lam, num_think_tokens,
                       num_answer_tokens, kept):
        if (sample_idx, lam) in self.completed_pairs:
            return None

        kl_loss, sparsity_loss, total_loss = compute_final_losses(
            model, base_embeddings, final_mask_hard, think_start, think_end,
            pred_positions, normal_probs, lambda_sparsity=lam,
        )

        dropped = num_think_tokens - kept
        kept_fraction = kept / num_think_tokens if num_think_tokens else 0.0

        row = {
            "sample_idx": sample_idx,
            "status": "",
            "num_prompt_tokens": think_start,
            "num_think_tokens": num_think_tokens,
            "num_answer_tokens": num_answer_tokens,
            "seq_len": base_embeddings.shape[1],
            "kept_tokens": kept,
            "dropped_tokens": dropped,
            "kept_fraction": kept_fraction,
            "final_kl_loss": kl_loss,
            "final_sparsity_loss": sparsity_loss,
            "final_total_loss": total_loss,
            "lambda_sparsity": lam,
        }
        self._write_row(row)
        return {"kl_loss": kl_loss, "sparsity_loss": sparsity_loss, "total_loss": total_loss}

    def close(self):
        self._file.close()

    def write_summary(self, summary_csv_path):
        # rebuilds summary_csv_path from scratch using whatever is currently
        # in per_sample_csv, same as the other checkers
        results = {}
        skipped_count = {}

        with open(self.per_sample_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lam = float(row["lambda_sparsity"])
                except (KeyError, ValueError, TypeError):
                    continue
                results.setdefault(lam, {"kept_fraction": [], "kl": [], "sparsity": [], "total": []})
                skipped_count.setdefault(lam, 0)

                if row.get("status"):
                    skipped_count[lam] += 1
                    continue

                if row.get("kept_fraction", "") not in ("", None):
                    results[lam]["kept_fraction"].append(float(row["kept_fraction"]))
                if row.get("final_kl_loss", "") not in ("", None):
                    results[lam]["kl"].append(float(row["final_kl_loss"]))
                if row.get("final_sparsity_loss", "") not in ("", None):
                    results[lam]["sparsity"].append(float(row["final_sparsity_loss"]))
                if row.get("final_total_loss", "") not in ("", None):
                    results[lam]["total"].append(float(row["final_total_loss"]))

        def mean_std(values):
            if len(values) == 0:
                return float("nan"), float("nan")
            if len(values) == 1:
                return values[0], 0.0
            return statistics.mean(values), statistics.stdev(values)

        summary_rows = []
        for lam in sorted(results.keys()):
            mean_kf, std_kf = mean_std(results[lam]["kept_fraction"])
            mean_kl, std_kl = mean_std(results[lam]["kl"])
            mean_sp, std_sp = mean_std(results[lam]["sparsity"])
            mean_tot, std_tot = mean_std(results[lam]["total"])
            summary_rows.append({
                "lambda": lam,
                "num_samples_used": len(results[lam]["kept_fraction"]),
                "num_samples_skipped": skipped_count[lam],
                "mean_kept_fraction": mean_kf,
                "std_kept_fraction": std_kf,
                "mean_final_kl_loss": mean_kl,
                "std_final_kl_loss": std_kl,
                "mean_final_sparsity_loss": mean_sp,
                "std_final_sparsity_loss": std_sp,
                "mean_final_total_loss": mean_tot,
                "std_final_total_loss": std_tot,
            })

        summary_fieldnames = [
            "lambda", "num_samples_used", "num_samples_skipped",
            "mean_kept_fraction", "std_kept_fraction",
            "mean_final_kl_loss", "std_final_kl_loss",
            "mean_final_sparsity_loss", "std_final_sparsity_loss",
            "mean_final_total_loss", "std_final_total_loss",
        ]
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_rows