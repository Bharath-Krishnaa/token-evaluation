# Saves the training data for the indicator (MiniCPM5): hidden states of the thinking tokens
# and their oracle mask label (1 = kept, 0 = dropped). One .pt file per sample.

import csv
import os

import torch


class IndicatorChecker:
    def __init__(self, manifest_csv, out_dir, hidden_layers=(-1,)):
        self.manifest_csv = manifest_csv
        self.out_dir = out_dir
        self.hidden_layers = list(hidden_layers)
        os.makedirs(out_dir, exist_ok=True)

        # rows already in the CSV are not run again
        self.completed_pairs = self.load_completed_pairs()

        file_is_new = not os.path.exists(manifest_csv)
        self.fieldnames = ["sample_idx", "lambda", "hidden_layer", "num_think_tokens",
                           "kept_tokens", "dropped_tokens", "skip_reason", "saved_path"]
        self.csv_file = open(manifest_csv, "a", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        if file_is_new:
            self.writer.writeheader()
            self.csv_file.flush()

    def load_completed_pairs(self):
        completed = set()
        if not os.path.exists(self.manifest_csv):
            return completed
        with open(self.manifest_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
        os.fsync(self.csv_file.fileno())  # save to disk right away
        self.completed_pairs.add((row["sample_idx"], row["lambda"]))

    def layers_str(self):
        # e.g. [-1, 10] -> "-1;10"
        return ";".join(str(l) for l in self.hidden_layers)

    def record_skip(self, sample_idx, lam, num_think_tokens, skip_reason):
        if (sample_idx, lam) in self.completed_pairs:
            return
        self.write_row({
            "sample_idx": sample_idx,
            "lambda": lam,
            "hidden_layer": self.layers_str(),
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

        # extra forward pass to get the hidden states of all layers
        with torch.no_grad():
            outputs = model(inputs_embeds=base_embeddings, output_hidden_states=True)
        all_hidden = outputs.hidden_states  # 0 = embeddings, -1 = last layer i.e, 24th layer in MiniCPM5

        labels = final_mask_hard[0].float().detach().cpu()

        # hidden states of the thinking tokens for each chosen layer
        features = {}
        for layer in self.hidden_layers:
            features[layer] = all_hidden[layer][0, think_start:think_end, :].float().detach().cpu()

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
            "hidden_layer": self.layers_str(),
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
        # summary is made from all rows in the manifest CSV
        kept_counts = {}
        total_counts = {}
        skipped_count = {}
        used_count = {}

        with open(self.manifest_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
            writer = csv.DictWriter(
                f, fieldnames=["lambda", "num_samples_used", "num_samples_skipped",
                               "total_tokens_collected", "fraction_kept"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_rows