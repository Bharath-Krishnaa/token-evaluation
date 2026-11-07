# Uses the trained CNN indicator to predict the mask (instead of training a mask)
# and checks if the final answer is still the same as the normal model's answer (MiniCPM5).
# By default runs on the test samples saved in the indicator checkpoint.

import argparse
import csv
import os
import re
import statistics

import torch

import minicpm_pruning_core as pruning_core
from train_indicator_cnn import CNNIndicator


def extract_final_number(text):
    # last number in the text, e.g. "the answer is 98." -> 98.0 is taken for comparison
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


def load_completed(out_csv):
    # sample indices already in the CSV
    completed = set()
    if not os.path.exists(out_csv):
        return completed
    with open(out_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(int(row["sample_idx"]))
            except (KeyError, ValueError, TypeError):
                continue
    return completed


def write_row(writer, csv_file, row):
    writer.writerow(row)
    csv_file.flush()
    os.fsync(csv_file.fileno())  # saves to disk 


def keep_think_close(mask):
    # <think> is in the prompt so only </think> (last position) is always kept
    mask = mask.clone()
    mask[:, -1] = 1.0
    return mask


def predict_mask(cnn_model, model, base_embeddings, think_start, think_end,
                 layer, feature_mean, feature_std, threshold, device):
    with torch.no_grad():
        outputs = model(inputs_embeds=base_embeddings, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]
        features = hidden[0, think_start:think_end, :].float()

        # same normalization as in training
        x = (features - feature_mean) / feature_std
        x = x.T  # (hidden_dim, num_think_tokens)
        logits = cnn_model(x)
        probs = torch.sigmoid(logits)
        mask = (probs >= threshold).float().unsqueeze(0)  # (1, num_think_tokens)
        mask = keep_think_close(mask)

    return mask


def generate_masked_answer(model, tokenizer, base_embeddings, mask, think_start, think_end,
                           device, max_new_tokens):
    # apply the mask then let the model generate the answer after </think>
    embeddings = base_embeddings.clone()
    embeddings[:, think_start:think_end, :] = (
        base_embeddings[:, think_start:think_end, :] * mask.unsqueeze(-1)
    )
    prompt_embeds = embeddings[:, :think_end, :]
    attention_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # with inputs_embeds generate() returns only the new tokens
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="openbmb/MiniCPM5-1B")
    parser.add_argument("--ckpt", required=True, help="indicator checkpoint file")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample_idxs", type=int, nargs="+", default=None,
                        help="samples to test, default is the test samples from the checkpoint")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=7168)
    parser.add_argument("--answer_max_new_tokens", type=int, default=7168)
    parser.add_argument("--min_think_tokens", type=int, default=3)
    parser.add_argument("--out_csv", default="minicpm5_indicator_consistency.csv")
    parser.add_argument("--summary_csv", default="minicpm5_indicator_consistency_summary.csv")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"loading indicator from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    cnn_model = CNNIndicator(ckpt["input_dim"], ckpt["channels"], ckpt["kernel_size"], ckpt["activation"]).to(device)
    cnn_model.load_state_dict(ckpt["state_dict"])
    cnn_model.eval()
    layer = ckpt["layer"]
    feature_mean = ckpt["feature_mean"].to(device)
    feature_std = ckpt["feature_std"].to(device)

    if args.sample_idxs is not None:
        sample_idxs = args.sample_idxs
    else:
        sample_idxs = ckpt["test_sample_idxs"]
    if args.max_samples is not None:
        sample_idxs = sample_idxs[:args.max_samples]
    print(f"testing {len(sample_idxs)} samples, layer={layer}, threshold={args.threshold}")

    print(f"loading model {args.model_name}")
    tokenizer, model = pruning_core.load_model_and_tokenizer(args.model_name, device=device)

    print("loading gsm8k")
    from datasets import load_dataset
    data = load_dataset("openai/gsm8k", "main", split="train")

    completed = load_completed(args.out_csv)
    if completed:
        print(f"{len(completed)} samples already in {args.out_csv}, skipping them")

    file_is_new = not os.path.exists(args.out_csv)
    fieldnames = ["sample_idx", "num_think_tokens", "kept_tokens", "dropped_tokens",
                  "skip_reason", "baseline_pred", "masked_pred", "consistent"]
    csv_file = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if file_is_new:
        writer.writeheader()
        csv_file.flush()

    try:
        for i, sample_idx in enumerate(sample_idxs):
            if sample_idx in completed:
                print(f"[{i + 1}/{len(sample_idxs)}] sample {sample_idx} already done, skipping")
                continue

            question, _ = pruning_core.get_sample(data, sample_idx)
            print(f"\n[{i + 1}/{len(sample_idxs)}] sample {sample_idx}")

            q_tokens, thinking_text, answer_text, truncated_by_cap = \
                pruning_core.generate_reasoning_and_answer(
                    tokenizer, model, question, device, max_new_tokens=args.max_new_tokens,
                    label=f"sample {sample_idx}")

            input_ids, think_start, think_end, num_think_tokens, num_answer_tokens = \
                pruning_core.make_input(tokenizer, q_tokens, thinking_text, answer_text, device)

            skip_reason = None
            if truncated_by_cap:
                skip_reason = "truncated_max_new_tokens"
            elif num_think_tokens <= args.min_think_tokens:
                skip_reason = "empty_thinking"
            elif num_answer_tokens == 0:
                skip_reason = "empty_answer"

            if skip_reason is not None:
                print(f"  skipped: {skip_reason} (think tokens = {num_think_tokens})")
                write_row(writer, csv_file, {
                    "sample_idx": sample_idx, "num_think_tokens": num_think_tokens,
                    "kept_tokens": "", "dropped_tokens": "", "skip_reason": skip_reason,
                    "baseline_pred": "", "masked_pred": "", "consistent": "",
                })
                continue

            try:
                embed_layer = model.get_input_embeddings()
                with torch.no_grad():
                    base_embeddings = embed_layer(input_ids)

                mask = predict_mask(cnn_model, model, base_embeddings, think_start, think_end,
                                    layer, feature_mean, feature_std, args.threshold, device)
                kept = int(mask.sum().item())
                print(f"  indicator kept {kept}/{num_think_tokens}")

                masked_text = generate_masked_answer(
                    model, tokenizer, base_embeddings, mask, think_start, think_end,
                    device, args.answer_max_new_tokens)

                baseline_pred = extract_final_number(answer_text)
                masked_pred = extract_final_number(masked_text)
                consistent = numbers_match(masked_pred, baseline_pred)
                print(f"  baseline={baseline_pred}  masked={masked_pred}  consistent={consistent}")

                if baseline_pred is None:
                    baseline_pred = ""
                if masked_pred is None:
                    masked_pred = ""

                write_row(writer, csv_file, {
                    "sample_idx": sample_idx, "num_think_tokens": num_think_tokens,
                    "kept_tokens": kept, "dropped_tokens": num_think_tokens - kept,
                    "skip_reason": "",
                    "baseline_pred": baseline_pred,
                    "masked_pred": masked_pred,
                    "consistent": int(consistent),
                })

            except torch.cuda.OutOfMemoryError as e:
                # long thinking can run out of memory and I skip this sample and go on
                print(f"  out of memory on sample {sample_idx}, skipping ({e})")
                torch.cuda.empty_cache()
                write_row(writer, csv_file, {
                    "sample_idx": sample_idx, "num_think_tokens": num_think_tokens,
                    "kept_tokens": "", "dropped_tokens": "", "skip_reason": "cuda_oom",
                    "baseline_pred": "", "masked_pred": "", "consistent": "",
                })
                continue
    finally:
        csv_file.close()

    # summary from all rows in the CSV
    consistent_vals = []
    skipped_count = 0
    with open(args.out_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("skip_reason"):
                skipped_count += 1
                continue
            val = row.get("consistent", "")
            if val not in ("", None):
                consistent_vals.append(int(val))

    if consistent_vals:
        consistency = statistics.mean(consistent_vals)
    else:
        consistency = float("nan")
    print(f"\nused {len(consistent_vals)} samples, skipped {skipped_count}")
    print(f"answer consistency with indicator mask: {consistency:.4f} ({consistency * 100:.1f}%)")

    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_samples_used", "num_samples_skipped", "consistency"])
        writer.writeheader()
        writer.writerow({
            "num_samples_used": len(consistent_vals),
            "num_samples_skipped": skipped_count,
            "consistency": consistency,
        })


if __name__ == "__main__":
    main()