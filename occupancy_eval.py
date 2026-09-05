"""
For each of the first N GSM8K samples, and for each lambda_sparsity value
in --lambdas:
  1) generate the model's own reasoning + answer ONCE per sample 
  2) train mask_param on the thinking tokens with that lambda
  3) freeze the trained mask into a hard 0/1 mask, run the model ONE more
     time with that hard mask applied

occupancy@k for one answer position:
    |top-k(unmasked) intersect top-k(masked)| / k
  averaged over all answer positions in a sample -> one number per
  (sample, lambda, k). 1.0 means the masked model's top-k exactly matches
  the unmasked model's top-k at every position; 0.0 means no overlap at all.

"""

import argparse
import csv
import os
import statistics

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def get_sample(data, idx):
    item = data[idx]
    return item["question"], item["answer"]


def generate_reasoning_and_answer(tokenizer, model, question, device, max_new_tokens, label=""):
    
    messages = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(prompt_ids)

    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0, prompt_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # truncated_by_cap = True ONLY when generation ran all the way to max_new_tokens 
    
    truncated_by_cap = False

    if "<think>" in generated_text and "</think>" in generated_text:
        _, _, after_open = generated_text.partition("<think>")
        thinking_text, _, answer_text = after_open.partition("</think>")
    else:
        # Either the model answered without ever opening a <think> block,
        # OR generation hit max_new_tokens
       
        truncated_by_cap = len(new_tokens) >= max_new_tokens
        reason = "generation hit --max_new_tokens (likely truncated mid-thought)" \
            if truncated_by_cap else "no <think>/</think> tags found in generated text"
        print(f"  [WARNING]{(' ' + label) if label else ''}: falling back to empty "
              f"thinking_text ({reason}, generated {len(new_tokens)} tokens). "
              f"This sample will be EXCLUDED from the occupancy stats.")
        thinking_text = ""
        answer_text = generated_text

    return prompt_ids[0], thinking_text.strip(), answer_text.strip(), truncated_by_cap


def make_input(tokenizer, q_tokens, thinking_text, answer_text, device):
    think_tokens = tokenizer("<think>" + thinking_text + "</think>",
                              return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    ans_tokens = tokenizer(answer_text,
                            return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

    all_tokens = torch.cat([q_tokens, think_tokens, ans_tokens])
    input_ids = all_tokens.unsqueeze(0).to(device)

    think_start = len(q_tokens)
    think_end = think_start + len(think_tokens)

    return input_ids, think_start, think_end, len(think_tokens), len(ans_tokens)


def compute_base(model, input_ids, think_end, device):
    # The unmasked ("normal") forward pass and its predicted distribution at every answer position computed once per sample
    embed_layer = model.get_input_embeddings()
    seq_len = input_ids.shape[1]
    pred_positions = torch.arange(think_end - 1, seq_len - 1, device=device)

    with torch.no_grad():
        base_embeddings = embed_layer(input_ids)
        normal_logits = model(inputs_embeds=base_embeddings).logits
        normal_probs = F.softmax(normal_logits[:, pred_positions, :], dim=-1)

    return base_embeddings, normal_probs, pred_positions


def train_mask(model, base_embeddings, think_start, think_end, normal_probs, pred_positions,
                device, lambda_sparsity, steps, lr, warmup_steps, init_score):
    
    batch_size = base_embeddings.shape[0]
    num_think_tokens = think_end - think_start

    init_logit = torch.logit(torch.tensor(init_score)).item()
    mask_param = torch.full(
        (batch_size, num_think_tokens), init_logit, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([mask_param], lr=lr)

    for step in range(steps):
        warmup_progress = min(1.0, step / max(1, warmup_steps))
        lam = lambda_sparsity * warmup_progress

        optimizer.zero_grad()

        mask_soft = torch.sigmoid(mask_param)
        mask_hard = (mask_soft > 0.5).float()
        mask = mask_hard + mask_soft - mask_soft.detach()  # STE

        embeddings = base_embeddings.clone()
        embeddings[:, think_start:think_end, :] = (
            base_embeddings[:, think_start:think_end, :] * mask.unsqueeze(-1)
        )

        masked_logits = model(inputs_embeds=embeddings).logits
        masked_log_probs = F.log_softmax(masked_logits[:, pred_positions, :], dim=-1)

        kl_per_token = F.kl_div(masked_log_probs, normal_probs, reduction="none").sum(dim=-1)
        kl_loss = kl_per_token.mean()
        sparsity_loss = mask.mean()
        total_loss = kl_loss + lam * sparsity_loss

        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_mask_soft = torch.sigmoid(mask_param)
        final_mask_hard = (final_mask_soft > 0.5).float()

    return final_mask_hard


def compute_occupancy(model, base_embeddings, final_mask_hard, think_start, think_end,
                       pred_positions, normal_probs, max_k):
    # One more forward pass with the FROZEN hard mask
    with torch.no_grad():
        embeddings = base_embeddings.clone()
        embeddings[:, think_start:think_end, :] = (
            base_embeddings[:, think_start:think_end, :] * final_mask_hard.unsqueeze(-1)
        )
        masked_logits = model(inputs_embeds=embeddings).logits
        masked_probs = F.softmax(masked_logits[:, pred_positions, :], dim=-1)  # (1, num_answer, V)

    num_answer_positions = normal_probs.shape[1]
    occupancy_per_k = {}

    for k in range(1, max_k + 1):
        normal_topk = torch.topk(normal_probs[0], k=k, dim=-1).indices   # (num_answer, k)
        masked_topk = torch.topk(masked_probs[0], k=k, dim=-1).indices   # (num_answer, k)

        overlaps = []
        for pos in range(num_answer_positions):
            normal_set = set(normal_topk[pos].tolist())
            masked_set = set(masked_topk[pos].tolist())
            intersection = len(normal_set & masked_set)
            overlaps.append(intersection / k)  # occupancy@k for this position

        occupancy_per_k[k] = sum(overlaps) / len(overlaps)  # averaged over the sample

    return occupancy_per_k


def load_completed_pairs(per_sample_csv_path):
    """Returns a set of (sample_idx, lambda) pairs that already have a row
    in per_sample_csv_path -- whether that row was a real result or a
    skip_reason row, either way it doesn't need to be redone."""
    completed = set()
    if not os.path.exists(per_sample_csv_path):
        return completed
    with open(per_sample_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((int(row["sample_idx"]), float(row["lambda"])))
            except (KeyError, ValueError, TypeError):
                continue  
    return completed


def write_summary_from_csv(per_sample_csv_path, summary_csv_path, max_k):
    """Recomputes summary_csv_path FROM SCRATCH by reading whatever rows currently 
    exist in per_sample_csv_path. This is what makes occupancy_summary.csv always reflect the FULL accumulated dataset"""
    results = {}        # lam -> {k: [occupancy@k values from valid rows]}
    skipped_count = {}  # lam -> count of skip_reason rows

    with open(per_sample_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lam = float(row["lambda"])
            except (KeyError, ValueError, TypeError):
                continue
            results.setdefault(lam, {k: [] for k in range(1, max_k + 1)})
            skipped_count.setdefault(lam, 0)

            if row.get("skip_reason"):
                skipped_count[lam] += 1
                continue

            for k in range(1, max_k + 1):
                val = row.get(f"occupancy@{k}", "")
                if val not in ("", None):
                    results[lam][k].append(float(val))

    summary_rows = []
    for lam in sorted(results.keys()):
        num_used = len(results[lam][1])
        num_skipped = skipped_count[lam]
        row = {"lambda": lam, "num_samples_used": num_used, "num_samples_skipped": num_skipped}
        for k in range(1, max_k + 1):
            values = results[lam][k]
            if len(values) == 0:
                mean_v, std_v = float("nan"), float("nan")
            elif len(values) == 1:
                mean_v, std_v = values[0], 0.0
            else:
                mean_v = statistics.mean(values)
                std_v = statistics.stdev(values)  # sample std (N-1 denominator)
            row[f"mean_occupancy@{k}"] = mean_v
            row[f"std_occupancy@{k}"] = std_v
        summary_rows.append(row)

    summary_fieldnames = ["lambda", "num_samples_used", "num_samples_skipped"] + \
        [f"mean_occupancy@{k}" for k in range(1, max_k + 1)] + \
        [f"std_occupancy@{k}" for k in range(1, max_k + 1)]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start_idx", type=int, default=0,
                         help="first GSM8K sample index to process")
    parser.add_argument("--end_idx", type=int, default=None,
                     help="last GSM8K sample index to process, INCLUSIVE. If given, "
                          "this overrides --num_samples.")
    parser.add_argument("--num_samples", type=int, default=100,
                         help="used only when --end_idx is not given: processes "
                              "--start_idx .. --start_idx + num_samples - 1")
    parser.add_argument("--lambdas", type=float, nargs="+", default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--init_score", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=5120)
    parser.add_argument("--min_think_tokens", type=int, default=3,
                         help="samples with num_think_tokens below this (i.e. essentially "
                              "empty <think></think>, which tokenizes to 2) are excluded "
                              "from the occupancy stats")
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument("--per_sample_csv", default="occupancy_per_sample.csv")
    parser.add_argument("--summary_csv", default="occupancy_summary.csv")
    args = parser.parse_args()

    if args.end_idx is not None:
        sample_indices = list(range(args.start_idx, args.end_idx + 1))
    else:
        sample_indices = list(range(args.start_idx, args.start_idx + args.num_samples))

    device = torch.device(args.device)

    # resume support: don't redo (sample_idx, lambda) pairs that a previous run 
    # (an earlier batch, or a rerun of this same batch after a dropped connection) already wrote to per_sample_csv
    completed_pairs = load_completed_pairs(args.per_sample_csv)
    if completed_pairs:
        print(f"found {len(completed_pairs)} already-completed (sample_idx, lambda) row(s) "
              f"in {args.per_sample_csv} -- these will be skipped, not redone.")

    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # freeze the model -- only mask_param ever trains

    print("loading gsm8k...")
    from datasets import load_dataset
    data = load_dataset("openai/gsm8k", "main", split="train")

    
    file_is_new = not os.path.exists(args.per_sample_csv)
    fieldnames = ["sample_idx", "lambda", "num_think_tokens", "kept_tokens", "dropped_tokens",
                  "skip_reason"] + [f"occupancy@{k}" for k in range(1, args.max_k + 1)]
    csv_file = open(args.per_sample_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if file_is_new:
        writer.writeheader()
        csv_file.flush()

    def write_row(row):
        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())  # force it to disk, not just the OS buffer
        completed_pairs.add((row["sample_idx"], row["lambda"]))

    this_run_skipped = []  # (sample_idx, reason, num_think_tokens)

    try:
        for i, sample_idx in enumerate(sample_indices):

            
            pending_lambdas = [lam for lam in args.lambdas
                                if (sample_idx, lam) not in completed_pairs]
            if not pending_lambdas:
                print(f"[{i + 1}/{len(sample_indices)}] sample_idx={sample_idx}: "
                      f"already completed for all lambdas, skipping")
                continue

            question, _ = get_sample(data, sample_idx)
            print(f"\n[{i + 1}/{len(sample_indices)}] sample_idx={sample_idx} "
                  f"(pending lambdas: {pending_lambdas})")

            # generate reasoning + answer ONCE per sample -- reused across lambdas
            q_tokens, thinking_text, answer_text, truncated_by_cap = generate_reasoning_and_answer(
                tokenizer, model, question, device, max_new_tokens=args.max_new_tokens,
                label=f"sample_idx={sample_idx}",
            )

            input_ids, think_start, think_end, num_think_tokens, num_answer_tokens = make_input(
                tokenizer, q_tokens, thinking_text, answer_text, device
            )

            skip_reason = None
            if truncated_by_cap:
                skip_reason = "truncated_max_new_tokens"
            elif num_think_tokens <= args.min_think_tokens:
                skip_reason = "empty_thinking"
            elif num_answer_tokens == 0:
                skip_reason = "empty_answer"

            if skip_reason is not None:
                print(f"  SKIPPED sample_idx={sample_idx}: {skip_reason} "
                      f"(num_think_tokens={num_think_tokens}) -- excluded from occupancy stats")
                this_run_skipped.append((sample_idx, skip_reason, num_think_tokens))
                for lam in pending_lambdas:
                    write_row({
                        "sample_idx": sample_idx,
                        "lambda": lam,
                        "num_think_tokens": num_think_tokens,
                        "kept_tokens": "",
                        "dropped_tokens": "",
                        "skip_reason": skip_reason,
                        **{f"occupancy@{k}": "" for k in range(1, args.max_k + 1)},
                    })
                continue

            # unmasked forward pass also reused across lambdas
            base_embeddings, normal_probs, pred_positions = compute_base(
                model, input_ids, think_end, device
            )

            for lam in pending_lambdas:
                print(f"  training mask for lambda={lam} ...")
                final_mask_hard = train_mask(
                    model, base_embeddings, think_start, think_end, normal_probs, pred_positions,
                    device, lambda_sparsity=lam, steps=args.steps, lr=args.lr,
                    warmup_steps=args.warmup_steps, init_score=args.init_score,
                )

                occupancy_per_k = compute_occupancy(
                    model, base_embeddings, final_mask_hard, think_start, think_end,
                    pred_positions, normal_probs, max_k=args.max_k,
                )

                kept = int(final_mask_hard.sum().item())
                row = {
                    "sample_idx": sample_idx,
                    "lambda": lam,
                    "num_think_tokens": num_think_tokens,
                    "kept_tokens": kept,
                    "dropped_tokens": num_think_tokens - kept,
                    "skip_reason": "",
                }
                for k in range(1, args.max_k + 1):
                    row[f"occupancy@{k}"] = occupancy_per_k[k]

                write_row(row)
                occ_str = ", ".join(f"k={k}:{occupancy_per_k[k]:.3f}" for k in range(1, args.max_k + 1))
                print(f"    kept {kept}/{num_think_tokens} | occupancy: {occ_str}")
    finally:
        csv_file.close()

    print(f"\nappended this run's results to {args.per_sample_csv}")

    
    summary_rows = write_summary_from_csv(args.per_sample_csv, args.summary_csv, args.max_k)
    print(f"wrote cumulative summary (from all rows in {args.per_sample_csv}) to {args.summary_csv}")

    print("\n" + "=" * 60)
    print(f"CUMULATIVE SUMMARY across ALL samples recorded so far in {args.per_sample_csv}")
    print("=" * 60)
    if this_run_skipped:
        print(f"{len(this_run_skipped)} sample(s) excluded in THIS run:")
        for sample_idx, reason, num_think_tokens in this_run_skipped:
            print(f"  sample_idx={sample_idx}: {reason} (num_think_tokens={num_think_tokens})")
    for row in summary_rows:
        print(f"\nlambda={row['lambda']}  (n_used={row['num_samples_used']}, "
              f"n_skipped={row['num_samples_skipped']})")
        for k in range(1, args.max_k + 1):
            print(f"  k={k}: mean={row[f'mean_occupancy@{k}']:.4f}  std={row[f'std_occupancy@{k}']:.4f}")


if __name__ == "__main__":
    main()