"""
The actual model logic (loading the model, generating a sample's reasoning + answer, training the mask)
lives in pruning_core.py.

This script parses args, sets up whichever checkers are enabled below,
loops over samples, and prints summaries at the end.

occupancy_checker / top1_hit_checker / accuracy_checker / indicator_checker
each record their own thing about the frozen mask and write their own CSV.
They're all optional -- comment any of them out below and the script keeps
running, it just won't produce that checker's output.


This script only runs one lambda_sparsity value at a time (--lambda_sparsity).
To try another lambda, run again with a different value and different CSV
paths so the results don't get mixed together.
"""

import argparse

import torch

import pruning_core_clone_2 as pruning_core

#import occupancy_checker
#import top1_hit_checker
#import accuracy_checker
import indicator_checker
#import total_loss_checker

# if a checker import above is commented out, set it to None so the
# "if X is not None" checks below all just work
if "occupancy_checker" not in globals():
    occupancy_checker = None
if "top1_hit_checker" not in globals():
    top1_hit_checker = None
if "accuracy_checker" not in globals():
    accuracy_checker = None
if "indicator_checker" not in globals():
    indicator_checker = None
if "total_loss_checker" not in globals():
    total_loss_checker = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--lambda_sparsity", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--init_score", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=7168)
    parser.add_argument("--min_think_tokens", type=int, default=3)
    parser.add_argument("--max_k", type=int, default=5)
    # occupancy stats (only used if occupancy_checker is imported above)
    parser.add_argument("--occupancy_per_sample_csv", default="occupancy_per_sample_01.csv")
    parser.add_argument("--occupancy_summary_csv", default="occupancy_summary_01.csv")
    # top1 hit stats (only used if top1_hit_checker is imported above)
    parser.add_argument("--top1_per_sample_csv", default="top1_hit_per_sample.csv")
    parser.add_argument("--top1_summary_csv", default="top1_hit_summary.csv")
    # accuracy checking (only used if accuracy_checker is imported above)
    parser.add_argument("--accuracy_per_sample_csv", default="test_accuracy.csv")
    parser.add_argument("--accuracy_summary_csv", default="test_accuracy_summary.csv")
    parser.add_argument("--answer_max_new_tokens", type=int, default=7168)
    # indicator training-data collection (only used if indicator_checker is imported above)
    parser.add_argument("--indicator_data_dir", default="indicator_data")
    parser.add_argument("--indicator_manifest_csv", default="indicator_manifest.csv")
    parser.add_argument("--indicator_summary_csv", default="indicator_summary.csv")
    parser.add_argument("--indicator_hidden_layers", type=int, nargs="+", default=[-1])
    # total loss stats (only used if total_loss_checker is imported above)
    parser.add_argument("--total_loss_per_sample_csv", default="result.csv")
    parser.add_argument("--total_loss_summary_csv", default="result_summary.csv")
    args = parser.parse_args()

    if args.end_idx is not None:
        sample_indices = list(range(args.start_idx, args.end_idx + 1))
    else:
        sample_indices = list(range(args.start_idx, args.start_idx + args.num_samples))

    device = torch.device(args.device)

    print("loading model...")
    tokenizer, model = pruning_core.load_model_and_tokenizer(device=device)

    print("loading gsm8k...")
    from datasets import load_dataset
    data = load_dataset("openai/gsm8k", "main", split="train")

    # each checker owns its own CSV(s) and resume state, only created if
    # its import above wasn't commented out
    occ_checker = None
    if occupancy_checker is not None:
        occ_checker = occupancy_checker.OccupancyChecker(
            per_sample_csv=args.occupancy_per_sample_csv, max_k=args.max_k,
        )

    top1_checker = None
    if top1_hit_checker is not None:
        top1_checker = top1_hit_checker.Top1HitChecker(
            per_sample_csv=args.top1_per_sample_csv,
            max_k=args.max_k,
        )

    acc_checker = None
    if accuracy_checker is not None:
        acc_checker = accuracy_checker.AccuracyChecker(
            tokenizer, per_sample_csv=args.accuracy_per_sample_csv,
            device=device, answer_max_new_tokens=args.answer_max_new_tokens,
        )

    indicator_chk = None
    if indicator_checker is not None:
        indicator_chk = indicator_checker.IndicatorChecker(
            manifest_csv=args.indicator_manifest_csv, out_dir=args.indicator_data_dir,
            hidden_layers=args.indicator_hidden_layers,
        )

    total_loss_chk = None
    if total_loss_checker is not None:
        total_loss_chk = total_loss_checker.TotalLossChecker(
            per_sample_csv=args.total_loss_per_sample_csv,
        )

    active_checkers = [c for c in (occ_checker, top1_checker, acc_checker, indicator_chk, total_loss_chk)
                        if c is not None]
    if not active_checkers:
        raise SystemExit(
            "No checkers active -- uncomment at least one of the occupancy_checker / "
            "top1_hit_checker / accuracy_checker / indicator_checker / total_loss_checker "
            "imports at the top of this file."
        )

    this_run_skipped = []  # (sample_idx, reason, num_think_tokens)

    lam = args.lambda_sparsity

    try:
        for i, sample_idx in enumerate(sample_indices):

            # still needs (re)computing if any active checker doesn't have
            # this (sample_idx, lam) row yet -- each checker's record_skip/
            # record_result is a no-op if it already has that pair
            already_done = all(checker.is_done(sample_idx, lam) for checker in active_checkers)
            if already_done:
                print(f"[{i + 1}/{len(sample_indices)}] sample_idx={sample_idx}: "
                      f"already completed for lambda={lam}, skipping")
                continue

            question, _ = pruning_core.get_sample(data, sample_idx)
            print(f"\n[{i + 1}/{len(sample_indices)}] sample_idx={sample_idx} (lambda={lam})")

            # generate reasoning + answer once per sample
            q_tokens, thinking_text, answer_text, truncated_by_cap = \
                pruning_core.generate_reasoning_and_answer(
                    tokenizer, model, question, device, max_new_tokens=args.max_new_tokens,
                    label=f"sample_idx={sample_idx}",
                )

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
                print(f"  SKIPPED sample_idx={sample_idx}: {skip_reason} "
                      f"(num_think_tokens={num_think_tokens}) -- excluded from occupancy stats")
                this_run_skipped.append((sample_idx, skip_reason, num_think_tokens))
                if occ_checker is not None:
                    occ_checker.record_skip(sample_idx, lam, num_think_tokens, skip_reason)
                if top1_checker is not None:
                    top1_checker.record_skip(sample_idx, lam, num_think_tokens, skip_reason)
                if acc_checker is not None:
                    acc_checker.record_skip(sample_idx, lam, num_think_tokens, skip_reason)
                if indicator_chk is not None:
                    indicator_chk.record_skip(sample_idx, lam, num_think_tokens, skip_reason)
                if total_loss_chk is not None:
                    total_loss_chk.record_skip(sample_idx, lam, num_think_tokens, skip_reason)
                continue

            try:
                # unmasked forward pass for this sample
                base_embeddings, normal_probs, pred_positions = pruning_core.compute_base(
                    model, input_ids, think_end, device
                )

                print(f"  training mask for lambda={lam} ...")
                final_mask_hard = pruning_core.train_mask(
                    model, base_embeddings, think_start, think_end, normal_probs, pred_positions,
                    device, lambda_sparsity=lam, steps=args.steps, lr=args.lr,
                    warmup_steps=args.warmup_steps, init_score=args.init_score,
                )

                kept = int(final_mask_hard.sum().item())
                print(f"    kept {kept}/{num_think_tokens}")

                # reuses the base_embeddings/final_mask_hard already computed
                # above, so this is just one extra forward pass, not a repeat
                # of generation or mask training
                if occ_checker is not None:
                    occupancy_per_k = occ_checker.record_result(
                        model, base_embeddings, final_mask_hard, think_start, think_end,
                        pred_positions, normal_probs,
                        sample_idx=sample_idx, lam=lam,
                        num_think_tokens=num_think_tokens, kept=kept,
                    )
                    if occupancy_per_k is not None:
                        occ_str = ", ".join(f"k={k}:{occupancy_per_k[k]:.3f}"
                                             for k in range(1, args.max_k + 1))
                        print(f"    occupancy: {occ_str}")

                if top1_checker is not None:
                    top1_checker.record_result(
                        model, base_embeddings, final_mask_hard, think_start, think_end,
                        pred_positions, normal_probs,
                        sample_idx=sample_idx, lam=lam,
                        num_think_tokens=num_think_tokens, kept=kept,
                    )

                # adds one extra generate() call to see what the model would
                # freely answer under the frozen mask, compared to the
                # unmasked model's own answer_text
                if acc_checker is not None:
                    acc_checker.record_result(
                        model, base_embeddings, final_mask_hard, think_start, think_end,
                        sample_idx=sample_idx, lam=lam, num_think_tokens=num_think_tokens,
                        kept=kept, baseline_answer_text=answer_text,
                    )

                # uses final_mask_hard as the label for every reasoning token
                # in this sample, paired with that token's hidden state, and
                # saves both to disk for train_indicator.py to train on later
                if indicator_chk is not None:
                    indicator_chk.record_result(
                        model, base_embeddings, final_mask_hard, think_start, think_end,
                        sample_idx=sample_idx, lam=lam, num_think_tokens=num_think_tokens,
                        kept=kept,
                    )

                # recomputes the KL / sparsity / total loss once at the end,
                # using the same formula train_mask uses during training,
                # just evaluated with the frozen final mask instead of every step
                if total_loss_chk is not None:
                    total_loss_chk.record_result(
                        model, base_embeddings, final_mask_hard, think_start, think_end,
                        pred_positions, normal_probs,
                        sample_idx=sample_idx, lam=lam,
                        num_think_tokens=num_think_tokens, num_answer_tokens=num_answer_tokens,
                        kept=kept,
                    )
            except torch.cuda.OutOfMemoryError as e:
                # one bad sample (usually a long <think> block) shouldn't
                # kill an unattended run -- log it as skipped and move on
                print(f"  [OOM] sample_idx={sample_idx} ran out of GPU memory during "
                      f"mask training/checking -- skipping and continuing. ({e})")
                torch.cuda.empty_cache()
                this_run_skipped.append((sample_idx, "cuda_oom", num_think_tokens))
                if occ_checker is not None:
                    occ_checker.record_skip(sample_idx, lam, num_think_tokens, "cuda_oom")
                if top1_checker is not None:
                    top1_checker.record_skip(sample_idx, lam, num_think_tokens, "cuda_oom")
                if acc_checker is not None:
                    acc_checker.record_skip(sample_idx, lam, num_think_tokens, "cuda_oom")
                if indicator_chk is not None:
                    indicator_chk.record_skip(sample_idx, lam, num_think_tokens, "cuda_oom")
                if total_loss_chk is not None:
                    total_loss_chk.record_skip(sample_idx, lam, num_think_tokens, "cuda_oom")
                continue
    finally:
        if occ_checker is not None:
            occ_checker.close()
        if top1_checker is not None:
            top1_checker.close()
        if acc_checker is not None:
            acc_checker.close()
        if indicator_chk is not None:
            indicator_chk.close()
        if total_loss_chk is not None:
            total_loss_chk.close()

    summary_rows = None
    if occ_checker is not None:
        print(f"\nappended this run's occupancy results to {args.occupancy_per_sample_csv}")
        summary_rows = occ_checker.write_summary(args.occupancy_summary_csv)
        print(f"wrote cumulative summary (from all rows in {args.occupancy_per_sample_csv}) to {args.occupancy_summary_csv}")

    top1_summary_rows = None
    if top1_checker is not None:
        top1_summary_rows = top1_checker.write_summary(args.top1_summary_csv)
        print(f"wrote cumulative top1-hit summary (from all rows in {args.top1_per_sample_csv}) "
              f"to {args.top1_summary_csv}")

    accuracy_summary_rows = None
    if acc_checker is not None:
        accuracy_summary_rows = acc_checker.write_summary(args.accuracy_summary_csv)
        print(f"wrote cumulative accuracy summary (from all rows in {args.accuracy_per_sample_csv}) "
              f"to {args.accuracy_summary_csv}")

    indicator_summary_rows = None
    if indicator_chk is not None:
        indicator_summary_rows = indicator_chk.write_summary(args.indicator_summary_csv)
        print(f"wrote cumulative indicator-dataset summary (from all rows in {args.indicator_manifest_csv}) "
              f"to {args.indicator_summary_csv}")

    total_loss_summary_rows = None
    if total_loss_chk is not None:
        total_loss_summary_rows = total_loss_chk.write_summary(args.total_loss_summary_csv)
        print(f"wrote cumulative total-loss summary (from all rows in {args.total_loss_per_sample_csv}) "
              f"to {args.total_loss_summary_csv}")

    if this_run_skipped:
        print("\n" + "=" * 60)
        print(f"{len(this_run_skipped)} sample(s) excluded in THIS run:")
        for sample_idx, reason, num_think_tokens in this_run_skipped:
            print(f"  sample_idx={sample_idx}: {reason} (num_think_tokens={num_think_tokens})")

    if summary_rows is not None:
        print("\n" + "=" * 60)
        print(f"CUMULATIVE OCCUPANCY SUMMARY across ALL samples recorded so far in {args.occupancy_per_sample_csv}")
        print("=" * 60)
        for row in summary_rows:
            print(f"\nlambda={row['lambda']}  (n_used={row['num_samples_used']}, "
                  f"n_skipped={row['num_samples_skipped']})")
            for k in range(1, args.max_k + 1):
                print(f"  k={k}: mean={row[f'mean_occupancy@{k}']:.4f}  std={row[f'std_occupancy@{k}']:.4f}")

    if top1_summary_rows is not None:
        print("\n" + "=" * 60)
        print(f"CUMULATIVE TOP1-HIT SUMMARY across ALL rows recorded so far in {args.top1_per_sample_csv}")
        print("=" * 60)
        for row in top1_summary_rows:
            print(f"\nlambda={row['lambda']}  (n_used={row['num_samples_used']}, "
                  f"n_skipped={row['num_samples_skipped']})")
            for k in range(1, args.max_k + 1):
                print(f"  k={k}: mean_top1_hit={row[f'mean_top1_hit@{k}']:.4f}  "
                      f"std_top1_hit={row[f'std_top1_hit@{k}']:.4f}")

    if accuracy_summary_rows is not None:
        print("\n" + "=" * 60)
        print(f"CUMULATIVE ACCURACY SUMMARY across ALL rows recorded so far in {args.accuracy_per_sample_csv}")
        print("=" * 60)
        for row in accuracy_summary_rows:
            print(f"lambda={row['lambda']}  (n_used={row['num_samples_used']}, "
                  f"n_skipped={row['num_samples_skipped']})  accuracy={row['accuracy']:.4f}")

    if indicator_summary_rows is not None:
        print("\n" + "=" * 60)
        print(f"CUMULATIVE INDICATOR-DATASET SUMMARY across ALL rows recorded so far in "
              f"{args.indicator_manifest_csv}")
        print("=" * 60)
        for row in indicator_summary_rows:
            print(f"lambda={row['lambda']}  (n_used={row['num_samples_used']}, "
                  f"n_skipped={row['num_samples_skipped']})  "
                  f"total_tokens_collected={row['total_tokens_collected']}  "
                  f"fraction_kept={row['fraction_kept']:.4f}")

    if total_loss_summary_rows is not None:
        print("\n" + "=" * 60)
        print(f"CUMULATIVE TOTAL-LOSS SUMMARY across ALL rows recorded so far in {args.total_loss_per_sample_csv}")
        print("=" * 60)
        for row in total_loss_summary_rows:
            print(f"lambda={row['lambda']}  (n_used={row['num_samples_used']}, "
                  f"n_skipped={row['num_samples_skipped']})  "
                  f"kept_fraction={row['mean_kept_fraction']:.4f}  "
                  f"kl={row['mean_final_kl_loss']:.4f}  total={row['mean_final_total_loss']:.4f}")


if __name__ == "__main__":
    main()