# Saves the training data for the indicator....
# the hidden states of the thinking tokens which comes from theunmasked forward pass on base_embeddings and not from the masked one.
# and their oracle mask label (1 = kept, 0 = dropped). One .pt file per sample. This is done for around 2000 samples in this experiment.

import csv
import os

import torch


class IndicatorChecker:
    def __init__(self, per_sample_csv, out_dir, hidden_layers=(-1,)):
        self.per_sample_csv = per_sample_csv
        self.out_dir = out_dir
        self.hidden_layers = list(hidden_layers)
        os.makedirs(out_dir, exist_ok=True)

        # rows already in the CSV are not run again
        self.completed_pairs = self.load_completed_pairs()

        file_is_new = not os.path.exists(per_sample_csv)
        self.fieldnames = ["sample_idx", "lambda", "hidden_layers", "num_think_tokens",
                           "kept_tokens", "dropped_tokens", "skip_reason", "saved_path"]
        self.csv_file = open(per_sample_csv, "a", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        if file_is_new:
            self.writer.writeheader()
            self.csv_file.flush()

    def load_completed_pairs(self):
        completed = set()
        if not os.path.exists(self.per_sample_csv):
            return completed
        with open(self.per_sample_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    completed.add((int(row["sample_idx"]), float(row["lambda"])))
                except (KeyError, ValueError, TypeError):
                    continue
        return completed

    def is_done(self, sample_idx, lam):
        return (sample_idx, lam) in self.completed_pairs

    def write_row(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())  # saves to disk 
        self.completed_pairs.add((row["sample_idx"], row["lambda"]))

    def layers_str(self):
        # e.g. [-1, 10] -> "-1,10"
        return ",".join(str(l) for l in self.hidden_layers)

    def record_skip(self, sample_idx, lam, num_think_tokens, skip_reason):
        if (sample_idx, lam) in self.completed_pairs:
            return
        self.write_row({
            "sample_idx": sample_idx,
            "lambda": lam,
            "hidden_layers": self.layers_str(),
            "num_think_tokens": num_think_tokens,
            "kept_tokens": "",
            "dropped_tokens": "",
            "skip_reason": skip_reason,
            "saved_path": "",
        })

    def record_result(self, model, base_embeddings, final_mask_hard, think_start, think_end,
                      sample_idx, lam, num_think_tokens, kept):
        if (sample_idx, lam) in self.completed_pairs:
            return None

        with torch.no_grad():
            outputs = model(inputs_embeds=base_embeddings, output_hidden_states=True)
            all_hidden = outputs.hidden_states  # 0 = embeddings, -1 = last layer, 28th layer in Qwen and 24th layer in MiniCPM

        # hidden states of the thinking tokens for each chosen layer
        features = {}
        for layer in self.hidden_layers:
            hidden = all_hidden[layer][0]
            features[layer] = hidden[think_start:think_end, :].float().cpu()

        labels = final_mask_hard[0].float().cpu()

        save_path = os.path.join(self.out_dir, f"sample_{sample_idx:04d}_lambda{lam}.pt")
        torch.save({
            "sample_idx": sample_idx,
            "lambda_val": lam,
            "hidden_layers": self.hidden_layers,
            "features": features,  # layer -> (num_think_tokens, hidden_dim)
            "labels": labels,
        }, save_path)

        self.write_row({
            "sample_idx": sample_idx,
            "lambda": lam,
            "hidden_layers": self.layers_str(),
            "num_think_tokens": num_think_tokens,
            "kept_tokens": kept,
            "dropped_tokens": num_think_tokens - kept,
            "skip_reason": "",
            "saved_path": save_path,
        })
        return save_path

    def close(self):
        self.csv_file.close()

    def write_summary(self, summary_csv_path):
        # final summary is made from all rows in the per-sample CSV
        kept_counts = {}
        total_counts = {}
        skipped_count = {}
        used_count = {}

        with open(self.per_sample_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    lam = float(row["lambda"])
                except (KeyError, ValueError, TypeError):
                    continue

                if lam not in kept_counts:
                    kept_counts[lam] = []
                    total_counts[lam] = []
                    skipped_count[lam] = 0
                    used_count[lam] = 0

                if row.get("skip_reason"):
                    skipped_count[lam] += 1
                    continue

                used_count[lam] += 1
                kept_val = row.get("kept_tokens", "")
                total_val = row.get("num_think_tokens", "")
                if kept_val not in ("", None) and total_val not in ("", None):
                    kept_counts[lam].append(int(kept_val))
                    total_counts[lam].append(int(total_val))

        summary_rows = []
        for lam in sorted(kept_counts.keys()):
            total_kept = sum(kept_counts[lam])
            total_tokens = sum(total_counts[lam])
            if total_tokens:
                fraction_kept = total_kept / total_tokens
            else:
                fraction_kept = float("nan")
            summary_rows.append({
                "lambda": lam,
                "num_samples_used": used_count[lam],
                "num_samples_skipped": skipped_count[lam],
                "total_tokens_collected": total_tokens,
                "fraction_kept": fraction_kept,
            })

        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["lambda", "num_samples_used", "num_samples_skipped",
                                                   "total_tokens_collected", "fraction_kept"])
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_rows