
import csv
import os

import torch


class IndicatorChecker:
    def __init__(self, manifest_csv, out_dir, hidden_layers=(-1,)):
        self.manifest_csv = manifest_csv
        self.out_dir = out_dir
        self.hidden_layers = list(hidden_layers)
        os.makedirs(out_dir, exist_ok=True)

        self.completed_pairs = self._load_completed_pairs()

        file_is_new = not os.path.exists(manifest_csv)
        self.fieldnames = ["sample_idx", "lambda", "hidden_layers", "num_think_tokens",
                            "kept_tokens", "dropped_tokens", "skip_reason", "saved_path"]
        self.csv_file = open(manifest_csv, "a", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        if file_is_new:
            self.writer.writeheader()
            self.csv_file.flush()

    def _load_completed_pairs(self):
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

    def _write_row(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())
        self.completed_pairs.add((row["sample_idx"], row["lambda"]))

    def _layers_str(self):
        return ",".join(str(l) for l in self.hidden_layers)

    def record_skip(self, sample_idx, lam, num_think_tokens, skip_reason):
        if (sample_idx, lam) in self.completed_pairs:
            return
        self._write_row({
            "sample_idx": sample_idx,
            "lambda": lam,
            "hidden_layers": self._layers_str(),
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
            all_hidden = outputs.hidden_states  # one entry per layer, index -1 = last

        features = {}
        for layer in self.hidden_layers:
            hidden = all_hidden[layer][0]
            features[layer] = hidden[think_start:think_end, :].float().detach().cpu()

        labels = final_mask_hard[0].float().detach().cpu()

        save_path = os.path.join(self.out_dir, f"sample_{sample_idx:04d}_lambda{lam}.pt")
        torch.save({
            "sample_idx": sample_idx,
            "lambda_val": lam,
            "hidden_layers": self.hidden_layers,
            "features": features,   # dict: layer -> (num_think_tokens, hidden_dim) tensor
            "labels": labels,
        }, save_path)

        self._write_row({
            "sample_idx": sample_idx,
            "lambda": lam,
            "hidden_layers": self._layers_str(),
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
                kept_counts.setdefault(lam, [])
                total_counts.setdefault(lam, [])
                skipped_count.setdefault(lam, 0)
                used_count.setdefault(lam, 0)

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
            summary_rows.append({
                "lambda": lam,
                "num_samples_used": used_count[lam],
                "num_samples_skipped": skipped_count[lam],
                "total_tokens_collected": total_tokens,
                "fraction_kept": (total_kept / total_tokens) if total_tokens else float("nan"),
            })

        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["lambda", "num_samples_used", "num_samples_skipped",
                               "total_tokens_collected", "fraction_kept"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_rows