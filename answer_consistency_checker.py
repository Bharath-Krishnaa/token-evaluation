# Checks if the final answer stays the same after masking the thinking tokens.

import csv
import os
import re
import statistics

import torch


def extract_final_number(text):
    # last number in the text, e.g. "the answer is 1,024." -> 1024.0 is taken
    if not text:
        return None
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if not matches:
        return None
    return float(matches[-1])


def numbers_match(a, b, tol=1e-4):
    # no number found means no match
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


class AccuracyChecker:
    def __init__(self, tokenizer, per_sample_csv, device, answer_max_new_tokens=256):
        self.tokenizer = tokenizer
        self.device = device
        self.answer_max_new_tokens = answer_max_new_tokens
        self.per_sample_csv = per_sample_csv

        # rows already in the CSV are not run again
        self.completed_pairs = self.load_completed_pairs()
        if self.completed_pairs:
            print(f"accuracy: {len(self.completed_pairs)} rows already in {per_sample_csv}, skipping them")

        file_is_new = not os.path.exists(per_sample_csv)
        self.fieldnames = ["sample_idx", "lambda", "num_think_tokens", "kept_tokens",
                           "dropped_tokens", "skip_reason", "baseline_pred", "masked_pred",
                           "consistent_with_baseline"]
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
        os.fsync(self.csv_file.fileno())  # saves to disk 
        self.completed_pairs.add((row["sample_idx"], row["lambda"]))

    def record_skip(self, sample_idx, lam, num_think_tokens, skip_reason):
        if (sample_idx, lam) in self.completed_pairs:
            return
        self.write_row({
            "sample_idx": sample_idx,
            "lambda": lam,
            "num_think_tokens": num_think_tokens,
            "kept_tokens": "",
            "dropped_tokens": "",
            "skip_reason": skip_reason,
            "baseline_pred": "",
            "masked_pred": "",
            "consistent_with_baseline": "",
        })

    def record_result(self, model, base_embeddings, final_mask_hard, think_start, think_end,
                      sample_idx, lam, num_think_tokens, kept, baseline_answer_text):
        # baseline is the normal model's own answer in essence the unmasked model's own answer and not the GSM8K label
        if (sample_idx, lam) in self.completed_pairs:
            return None

        baseline_pred = extract_final_number(baseline_answer_text)
        masked_text = self.generate_masked_answer(
            model, base_embeddings, final_mask_hard, think_start, think_end
        )
        masked_pred = extract_final_number(masked_text)
        consistent = numbers_match(masked_pred, baseline_pred)

        if baseline_pred is None:
            baseline_pred = ""
        if masked_pred is None:
            masked_pred = ""

        self.write_row({
            "sample_idx": sample_idx,
            "lambda": lam,
            "num_think_tokens": num_think_tokens,
            "kept_tokens": kept,
            "dropped_tokens": num_think_tokens - kept,
            "skip_reason": "",
            "baseline_pred": baseline_pred,
            "masked_pred": masked_pred,
            "consistent_with_baseline": int(consistent),
        })
        return consistent

    def generate_masked_answer(self, model, base_embeddings, final_mask_hard,
                               think_start, think_end):
        # apply the mask and then let the model generate the answer after </think>
        embeddings = base_embeddings.clone()
        embeddings[:, think_start:think_end, :] = (
            base_embeddings[:, think_start:think_end, :] * final_mask_hard.unsqueeze(-1)
        )
        prompt_embeds = embeddings[:, :think_end, :]
        attention_mask = torch.ones(prompt_embeds.shape[:2], device=self.device, dtype=torch.long)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.answer_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # with inputs_embeds generate() returns only the new tokens
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    def close(self):
        self.csv_file.close()

    def write_summary(self, summary_csv_path):
        # final summary is made from all rows in the per sample CSV
        consistent_vals = {}
        skipped_count = {}
        used_count = {}

        with open(self.per_sample_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lam = float(row["lambda"])
                except (KeyError, ValueError, TypeError):
                    continue

                if lam not in consistent_vals:
                    consistent_vals[lam] = []
                    skipped_count[lam] = 0
                    used_count[lam] = 0

                if row.get("skip_reason"):
                    skipped_count[lam] += 1
                    continue

                used_count[lam] += 1
                val = row.get("consistent_with_baseline", "")
                if val not in ("", None):
                    consistent_vals[lam].append(int(val))

        summary_rows = []
        for lam in sorted(consistent_vals.keys()):
            # fraction of samples where the masked answer matches the normal answer
            if consistent_vals[lam]:
                accuracy = statistics.mean(consistent_vals[lam])
            else:
                accuracy = float("nan")
            summary_rows.append({
                "lambda": lam,
                "num_samples_used": used_count[lam],
                "num_samples_skipped": skipped_count[lam],
                "accuracy": accuracy,
            })

        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["lambda", "num_samples_used", "num_samples_skipped", "accuracy"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_rows