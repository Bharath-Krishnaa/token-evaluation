# Runs pruning on GSM8K samples and saves the results to CSV files.
# Comment out a checker import below if you don't need it.

import argparse

import torch

import pruning_core

# checkers that are not imported stay None
occupancy_checker = None
top1_hit_checker = None
accuracy_checker = None
indicator_checker = None
total_loss_checker = None

import occupancy_checker
import top1_hit_checker
import answer_consistency_checker as accuracy_checker
import indicator_checker
#import total_loss_checker


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
    # occupancy
    parser.add_argument("--occupancy_per_sample_csv", default="occupancy_per_sample_01.csv")
    parser.add_argument("--occupancy_summary_csv", default="occupancy_summary_01.csv")
    # top1 hit
    parser.add_argument("--top1_per_sample_csv", default="top1_hit_per_sample.csv")
    parser.add_argument("--top1_summary_csv", default="top1_hit_summary.csv")
    # accuracy
    parser.add_argument("--accuracy_per_sample_csv", default="test_accuracy.csv")
    parser.add_argument("--accuracy_summary_csv", default="test_accuracy_summary.csv")
    parser.add_argument("--answer_max_new_tokens", type=int, default=7168)
    # indicator data
    parser.add_argument("--indicator_data_dir", default="indicator_data")
    parser.add_argument("--indicator_csv", default="indicator.csv")
    parser.add_argument("--indicator_summary_csv", default="indicator_summary.csv")
    parser.add_argument("--indicator_hidden_layers", type=int, nargs="+", default=[-1])
    # total loss
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

    # create only the checkers that are imported
    occ_checker = None
    if occupancy_checker is not None:
        occ_checker = occupancy_checker.OccupancyChecker(
            per_sample_csv=args.occupancy_per_sample_csv, max_k=args.max_k,
        )

    top1_checker = None
    if top1_hit_checker is not None:
        top1_checker = top1_hit_checker.Top1HitChecker(
            per_sample_csv=args.top1_per_sample_csv, max_k=args.max_k,
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
            per_sample_csv=args.indicator_csv, out_dir=args.indicator_data_dir,
            hidden_layers=args.indicator_hidden_layers,
        )

    total_loss_chk = None
    if total_loss_checker is not None:
        total_loss_chk = total_loss_checker.TotalLossChecker(
            per_sample_csv=args.total_loss_per_sample_csv,
        )

    active_checkers = []
    for c in [occ_checker, top1_checker, acc_checker, indicator_chk, total_loss_chk]:
        if c is not None:
            active_checkers.append(c)

    if len(active_checkers) == 0:
        raise SystemExit("No checker is imported, uncomment at least one import at the top.")

    skipped = []  # (sample_idx, reason, num_think_tokens)

    lam = args.lambda_sparsity

    try:
        for i, sample_idx in enumerate(sample_indices):

            # skip if every checker already has this sample for this lambda
            already_done = all(checker.is_done(sample_idx, lam) for checker in active_checkers)
            if already_done:
                print(f"[{i + 1}/{len(sample_indices)}] sample {sample_idx} already done, skipping")
                continue

            question, _ = pruning_core.get_sample(data, sample_idx)
            print(f"\n[{i + 1}/{len(sample_indices)}] sample {sample_idx} (lambda={lam})")

            q_tokens, thinking_text, answer_text, truncated_by_cap = \
                pruning_core.generate_reasoning_and_answer(
                    tokenizer, model, question, device, max_new_tokens=args.max_new_tokens,
                    label=f"sample {sample_idx}",
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
                print(f"  skipped sample {sample_idx}: {skip_reason} (think tokens = {num_think_tokens})")
                skipped.append((sample_idx, skip_reason, num_think_tokens))
                for checker in active_checkers:
                    checker.record_skip(sample_idx, lam, num_think_tokens, skip_reason)
                continue

            try:
                base_embeddings, normal_probs, pred_positions = pruning_core.compute_base(
                    model, input_ids, think_end, device
                )

                print(f"  training mask, lambda={lam}")
                final_mask_hard = pruning_core.train_mask(
                    model, base_embeddings, think_start, think_end, normal_probs, pred_positions,
                    device, lambda_sparsity=lam, steps=args.steps, lr=args.lr,
                    warmup_steps=args.warmup_steps, init_score=args.init_score,
                )

                kept = int(final_mask_hard.sum().item())
                print(f"    kept {kept}/{num_think_tokens}")

                # all checkers use the same final mask

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

                # generates the answer with the mask and compares with the normal answer
                if acc_checker is not None:
                    acc_checker.record_result(
                        model, base_embeddings, final_mask_hard, think_start, think_end,
                        sample_idx=sample_idx, lam=lam, num_think_tokens=num_think_tokens,
                        kept=kept, baseline_answer_text=answer_text,
                    )

                # saves hidden states + mask labels for training the indicator
                if indicator_chk is not None:
                    indicator_chk.record_result(
                        model, base_embeddings, final_mask_hard, think_start, think_end,
                        sample_idx=sample_idx, lam=lam, num_think_tokens=num_think_tokens,
                        kept=kept,
                    )

                # KL / sparsity / total loss with the final mask
                if total_loss_chk is not None:
                    total_loss_chk.record_result(
                        model, base_embeddings, final_mask_hard, think_start, think_end,
                        pred_positions, normal_probs,
                        sample_idx=sample_idx, lam=lam,
                        num_think_tokens=num_think_tokens, num_answer_tokens=num_answer_tokens,
                        kept=kept,
                    )
            except torch.cuda.OutOfMemoryError as e:
                # long thinking can run out of memory, skip this sample and go on
                print(f"  out of memory on sample {sample_idx}, skipping ({e})")
                torch.cuda.empty_cache()
                skipped.append((sample_idx, "cuda_oom", num_think_tokens))
                for checker in active_checkers:
                    checker.record_skip(sample_idx, lam, num_think_tokens, "cuda_oom")
                continue
    finally:
        for checker in active_checkers:
            checker.close()

    # summaries are computed from all rows in the CSVs, not only this run

    summary_rows = None
    if occ_checker is not None:
        summary_rows = occ_checker.write_summary(args.occupancy_summary_csv)
        print(f"\nwrote occupancy summary to {args.occupancy_summary_csv}")

    top1_summary_rows = None
    if top1_checker is not None:
        top1_summary_rows = top1_checker.write_summary(args.top1_summary_csv)
        print(f"wrote top1 hit summary to {args.top1_summary_csv}")

    accuracy_summary_rows = None
    if acc_checker is not None:
        accuracy_summary_rows = acc_checker.write_summary(args.accuracy_summary_csv)
        print(f"wrote accuracy summary to {args.accuracy_summary_csv}")

    indicator_summary_rows = None
    if indicator_chk is not None:
        indicator_summary_rows = indicator_chk.write_summary(args.indicator_summary_csv)
        print(f"wrote indicator summary to {args.indicator_summary_csv}")

    total_loss_summary_rows = None
    if total_loss_chk is not None:
        total_loss_summary_rows = total_loss_chk.write_summary(args.total_loss_summary_csv)
        print(f"wrote total loss summary to {args.total_loss_summary_csv}")

    if skipped:
        print(f"\nskipped {len(skipped)} sample(s) in this run:")
        for sample_idx, reason, num_think_tokens in skipped:
            print(f"  sample {sample_idx}: {reason} (think tokens = {num_think_tokens})")

    if summary_rows is not None:
        print("\noccupancy summary")
        for row in summary_rows:
            print(f"lambda={row['lambda']}  used={row['num_samples_used']}  "
                  f"skipped={row['num_samples_skipped']}")
            for k in range(1, args.max_k + 1):
                print(f"  k={k}: mean={row[f'mean_occupancy@{k}']:.4f}  std={row[f'std_occupancy@{k}']:.4f}")

    if top1_summary_rows is not None:
        print("\ntop1 hit summary")
        for row in top1_summary_rows:
            print(f"lambda={row['lambda']}  used={row['num_samples_used']}  "
                  f"skipped={row['num_samples_skipped']}")
            for k in range(1, args.max_k + 1):
                print(f"  k={k}: mean={row[f'mean_top1_hit@{k}']:.4f}  std={row[f'std_top1_hit@{k}']:.4f}")

    if accuracy_summary_rows is not None:
        print("\naccuracy summary")
        for row in accuracy_summary_rows:
            print(f"lambda={row['lambda']}  used={row['num_samples_used']}  "
                  f"skipped={row['num_samples_skipped']}  accuracy={row['accuracy']:.4f}")

    if indicator_summary_rows is not None:
        print("\nindicator summary")
        for row in indicator_summary_rows:
            print(f"lambda={row['lambda']}  used={row['num_samples_used']}  "
                  f"skipped={row['num_samples_skipped']}  "
                  f"tokens={row['total_tokens_collected']}  kept={row['fraction_kept']:.4f}")

    if total_loss_summary_rows is not None:
        print("\ntotal loss summary")
        for row in total_loss_summary_rows:
            print(f"lambda={row['lambda']}  used={row['num_samples_used']}  "
                  f"skipped={row['num_samples_skipped']}  "
                  f"kept={row['mean_kept_fraction']:.4f}  "
                  f"kl={row['mean_final_kl_loss']:.4f}  total={row['mean_final_total_loss']:.4f}")


if __name__ == "__main__":
    main()