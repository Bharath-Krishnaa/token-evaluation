"""
Checks whether masking changes the model's FINAL ANSWER, not just its
next-token distributions.

"""

import csv
import os
import re
import statistics

import torch


def extract_final_number(text):
    # pulls out the last number in text, e.g. "...answer is 42." -> 42.0
    # strips commas first so "1,024" parses as 1024
    if not text:
        return None
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def numbers_match(a, b, tol=1e-4):
    # None means "no answer extracted" -- never counts as matching
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


class AccuracyChecker:
    def __init__(self, tokenizer, per_sample_csv, device, answer_max_new_tokens=256):
        self.tokenizer = tokenizer
        self.device = device
        self.answer_max_new_tokens = answer_max_new_tokens
        self.per_sample_csv = per_sample_csv

        self.completed_pairs = self._load_completed_pairs()
        if self.completed_pairs:
            print(f"[accuracy_checker] found {len(self.completed_pairs)} already-completed "
                  f"(sample_idx, lambda) row(s) in {per_sample_csv}, skipping those")

        file_is_new = not os.path.exists(per_sample_csv)
        self.fieldnames = ["sample_idx", "lambda", "num_think_tokens", "kept_tokens",
                            "dropped_tokens", "skip_reason", "baseline_pred", "masked_pred",
                            "consistent_with_baseline"]
        self.csv_file = open(per_sample_csv, "a", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        if file_is_new:
            self.writer.writeheader()
            self.csv_file.flush()

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
        self.writer.writerow(row)
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())  # write to disk right away
        self.completed_pairs.add((row["sample_idx"], row["lambda"]))

    def record_skip(self, sample_idx, lam, num_think_tokens, skip_reason):
        if (sample_idx, lam) in self.completed_pairs:
            return
        self._write_row({
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
        # baseline_answer_text is the UNMASKED model's own generated answer
        # (the answer_text already computed in the main script) -- the
        # reference point here, not GSM8K's ground truth.
        # Reuses the same base_embeddings/final_mask_hard already computed,
        # so this only adds one extra generate() call.
        if (sample_idx, lam) in self.completed_pairs:
            return None

        baseline_pred = extract_final_number(baseline_answer_text)
        masked_generated_text = self._generate_masked_answer(
            model, base_embeddings, final_mask_hard, think_start, think_end
        )
        masked_pred = extract_final_number(masked_generated_text)
        consistent = numbers_match(masked_pred, baseline_pred)

        self._write_row({
            "sample_idx": sample_idx,
            "lambda": lam,
            "num_think_tokens": num_think_tokens,
            "kept_tokens": kept,
            "dropped_tokens": num_think_tokens - kept,
            "skip_reason": "",
            "baseline_pred": "" if baseline_pred is None else baseline_pred,
            "masked_pred": "" if masked_pred is None else masked_pred,
            "consistent_with_baseline": int(consistent),
        })
        return consistent

    def _generate_masked_answer(self, model, base_embeddings, final_mask_hard,
                                 think_start, think_end):
        # free generation under the frozen mask, no teacher forcing. When
        # generate() is called with inputs_embeds instead of input_ids, the
        # returned tensor is ONLY the newly generated tokens, so there's no
        # prompt-length slicing needed here.
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

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    def close(self):
        self.csv_file.close()

    def write_summary(self, summary_csv_path):
        # rebuilds summary_csv_path from scratch using whatever is currently
        # in per_sample_csv, so it reflects the full accumulated dataset
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
                consistent_vals.setdefault(lam, [])
                skipped_count.setdefault(lam, 0)
                used_count.setdefault(lam, 0)

                if row.get("skip_reason"):
                    skipped_count[lam] += 1
                    continue

                used_count[lam] += 1
                val = row.get("consistent_with_baseline", "")
                if val not in ("", None):
                    consistent_vals[lam].append(int(val))

        summary_rows = []
        for lam in sorted(consistent_vals.keys()):
            summary_rows.append({
                "lambda": lam,
                "num_samples_used": used_count[lam],
                "num_samples_skipped": skipped_count[lam],
                # fraction of used samples where the freely regenerated
                # masked answer matched the unmasked model's original answer
                "accuracy": (statistics.mean(consistent_vals[lam])
                             if consistent_vals[lam] else float("nan")),
            })

        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["lambda", "num_samples_used", "num_samples_skipped", "accuracy"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        return summary_rows