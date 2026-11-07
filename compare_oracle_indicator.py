# Compares answer consistency of the oracle mask and the indicator mask
# on the indicator's test samples.
# oracle csv: from pruning_eval.py, indicator csv: from indicator_consistency_checker.py

import argparse
import csv

import torch


def read_rows(path, consistent_col):
    # reads non skipped rows: sample_idx -> consistent, kept, total
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("skip_reason"):
                continue
            if r.get(consistent_col, "") in ("", None):
                continue
            rows[int(r["sample_idx"])] = {
                "consistent": int(r[consistent_col]),
                "kept": int(r["kept_tokens"]),
                "total": int(r["num_think_tokens"]),
            }
    return rows


def summarize(rows):
    # returns number of samples, fraction of tokens kept, consistency
    n = len(rows)
    if n == 0:
        return 0, float("nan"), float("nan")
    kept = sum(r["kept"] for r in rows.values())
    total = sum(r["total"] for r in rows.values())
    cons = sum(r["consistent"] for r in rows.values()) / n
    return n, kept / total, cons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--oracle_csv", required=True)
    parser.add_argument("--indicator_csv", required=True)
    parser.add_argument("--oracle_col", default=None,
                        help="consistency column in the oracle csv, found from the header if not given")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    test_idxs = set(ckpt["test_sample_idxs"])
    print(f"checkpoint has {len(test_idxs)} test samples (layer {ckpt['layer']})")

    # baseline = answer of Qwen3/MiniCPM with no mask (all thinking tokens kept)
    # oracle csv: consistent_with_baseline = 1 if the oracle mask answer equals the baseline answer
    # indicator csv: consistent = 1 if the indicator mask answer equals the baseline answer
    # only the indicator's test samples are used and we compare these two consistency rates
    # older oracle csv files calls this column as "same" so both are same meaning
    oracle_col = args.oracle_col
    if oracle_col is None:
        with open(args.oracle_csv, newline="") as f:
            header = next(csv.reader(f))
        if "consistent_with_baseline" in header:
            oracle_col = "consistent_with_baseline"
        else:
            oracle_col = "same"
    print(f"oracle consistency column: {oracle_col}")

    # keep only the indicator's test samples
    oracle = {}
    for i, r in read_rows(args.oracle_csv, oracle_col).items():
        if i in test_idxs:
            oracle[i] = r

    indicator = {}
    for i, r in read_rows(args.indicator_csv, "consistent").items():
        if i in test_idxs:
            indicator[i] = r

    missing = test_idxs - set(oracle)
    if missing:
        print(f"{len(missing)} test samples have no oracle result (skipped or not run)")

    # samples that have a result in both files
    common = sorted(set(oracle) & set(indicator))
    oracle_c = {i: oracle[i] for i in common}
    indicator_c = {i: indicator[i] for i in common}

    print("\nall test samples with a result")
    for name, rows in [("oracle", oracle), ("indicator", indicator)]:
        n, fk, cons = summarize(rows)
        print(f"  {name:9s}  n={n:4d}  fraction_kept={fk:.4f}  consistency={cons:.4f}")

    print(f"\nsamples with a result in both: n={len(common)}")
    rows_out = []
    for name, rows in [("oracle", oracle_c), ("indicator", indicator_c)]:
        n, fk, cons = summarize(rows)
        print(f"  {name:9s}  fraction_kept={fk:.4f}  consistency={cons:.4f}")
        rows_out.append({"method": name, "n": n, "fraction_kept": fk, "consistency": cons})

    # count how often each method gets the same answer on the common samples
    both = 0
    only_oracle = 0
    only_ind = 0
    for i in common:
        o = oracle[i]["consistent"]
        d = indicator[i]["consistent"]
        if o and d:
            both += 1
        elif o and not d:
            only_oracle += 1
        elif not o and d:
            only_ind += 1
    neither = len(common) - both - only_oracle - only_ind

    print(f"\n  both consistent:            {both}")
    print(f"  only oracle consistent:     {only_oracle}")
    print(f"  only indicator consistent:  {only_ind}")
    print(f"  neither:                    {neither}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "n", "fraction_kept", "consistency",
                                              "both", "only_oracle", "only_indicator", "neither"])
            w.writeheader()
            for r in rows_out:
                r["both"] = both
                r["only_oracle"] = only_oracle
                r["only_indicator"] = only_ind
                r["neither"] = neither
                w.writerow(r)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()