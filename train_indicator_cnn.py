
import argparse
import glob
import os
import random
import torch
import torch.nn as nn


def load_dataset(data_dir, lambda_val):
    files = sorted(glob.glob(os.path.join(data_dir, f"sample_*_lambda{lambda_val}.pt")))
    if not files:
        raise FileNotFoundError(f"no files matching sample_*_lambda{lambda_val}.pt in {data_dir}")
    return [torch.load(fp) for fp in files]


def split_samples(samples, val_frac, test_frac, seed):
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    n = len(idxs)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    test_idxs = set(idxs[:n_test])
    val_idxs = set(idxs[n_test:n_test + n_val])
    train_idxs = set(idxs[n_test + n_val:])
    train = [samples[i] for i in sorted(train_idxs)]
    val = [samples[i] for i in sorted(val_idxs)]
    test = [samples[i] for i in sorted(test_idxs)]
    return train, val, test


def feature_stats(samples, layer):
    feats = torch.cat([s["features"][layer] for s in samples], dim=0).float()
    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


class CNNIndicator(nn.Module):
    def __init__(self, input_dim, channels, kernel_size):
        super().__init__()
        layers = []
        prev = input_dim
        for c in channels:
            layers += [nn.Conv1d(prev, c, kernel_size, padding=kernel_size // 2), nn.ReLU()]
            prev = c
        layers += [nn.Conv1d(prev, 1, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (hidden_dim, seq_len) -- conv1d wants (batch, channels, length)
        out = self.net(x.unsqueeze(0))
        return out.squeeze(0).squeeze(0)  # back to (seq_len,)


def run_epoch(model, samples, layer, mean, std, device, optimizer, criterion):
    training = optimizer is not None
    model.train() if training else model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for s in samples:
            x = s["features"][layer].float().to(device)
            x = (x - mean) / std
            x = x.T
            y = s["labels"].float().to(device)

            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(y.detach().cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    return total_loss / len(all_labels), all_probs, all_labels


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).float()
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    n = len(y_true)
    accuracy = (tp + tn) / n if n else 0.0

    # class 1 = kept
    precision_kept = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_kept = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_kept = 2 * precision_kept * recall_kept / (precision_kept + recall_kept) \
        if (precision_kept + recall_kept) > 0 else 0.0

    # class 0 = dropped -- the class that actually matters for pruning
    precision_drop = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_drop = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_drop = 2 * precision_drop * recall_drop / (precision_drop + recall_drop) \
        if (precision_drop + recall_drop) > 0 else 0.0

    macro_f1 = (f1_kept + f1_drop) / 2

    base_rate = y_true.mean().item()
    majority_frac = max(base_rate, 1 - base_rate)
    majority_f1 = 2 * majority_frac / (1 + majority_frac)
    majority_baseline_macro_f1 = majority_f1 / 2  # minority class F1 is always 0 for a majority-only baseline

    return dict(
        accuracy=accuracy,
        precision_kept=precision_kept, recall_kept=recall_kept, f1_kept=f1_kept,
        precision_drop=precision_drop, recall_drop=recall_drop, f1_drop=f1_drop,
        macro_f1=macro_f1,
        tp=tp, tn=tn, fp=fp, fn=fn, n=n, base_rate=base_rate,
        majority_baseline=majority_frac,
        majority_baseline_macro_f1=majority_baseline_macro_f1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="indicator_data")
    parser.add_argument("--lambda_val", type=float, default=0.01)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--channels", default="64,32",
                         help="comma separated conv channel sizes, e.g. 64,32")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out_ckpt", default="indicator_model_cnn.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    channels = [int(x) for x in args.channels.split(",") if x.strip()]

    print(f"loading dataset from {args.data_dir} (lambda={args.lambda_val}, layer={args.layer}) ...")
    samples = load_dataset(args.data_dir, args.lambda_val)
    print(f"found {len(samples)} sample files")

    for s in samples:
        if args.layer not in s["hidden_layers"]:
            raise ValueError(
                f"sample_idx={s['sample_idx']} does not have layer {args.layer} saved "
                f"(it has {s['hidden_layers']})"
            )

    train_s, val_s, test_s = split_samples(samples, args.val_frac, args.test_frac, args.seed)
    print(f"split: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test samples")

    mean, std = feature_stats(train_s, args.layer)
    mean, std = mean.to(device), std.to(device)

    train_labels = torch.cat([s["labels"] for s in train_s])
    pos_frac = train_labels.float().mean().clamp(1e-6, 1 - 1e-6)
    pos_weight = torch.tensor([(1 - pos_frac) / pos_frac], device=device)
    print(f"fraction kept in train set: {pos_frac.item():.3f}")

    input_dim = train_s[0]["features"][args.layer].shape[1]
    model = CNNIndicator(input_dim, channels, args.kernel_size).to(device)
    print(f"model: {model}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_macro_f1 = -1.0
    best_state = None
    for epoch in range(args.epochs):
        train_loss, _, _ = run_epoch(model, train_s, args.layer, mean, std, device, optimizer, criterion)
        _, val_probs, val_labels = run_epoch(model, val_s, args.layer, mean, std, device, None, criterion)
        val_metrics = binary_metrics(val_labels, val_probs)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"val_f1_kept={val_metrics['f1_kept']:.4f}  val_f1_drop={val_metrics['f1_drop']:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")

    model.load_state_dict(best_state)
    _, test_probs, test_labels = run_epoch(model, test_s, args.layer, mean, std, device, None, criterion)
    test_metrics = binary_metrics(test_labels, test_probs)

    print("\ntest set results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    print(f"\nmajority baseline accuracy:  {test_metrics['majority_baseline']:.4f}")
    print(f"indicator accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"majority baseline macro F1:  {test_metrics['majority_baseline_macro_f1']:.4f}")
    print(f"indicator macro F1:          {test_metrics['macro_f1']:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "channels": channels,
        "kernel_size": args.kernel_size,
        "input_dim": input_dim,
        "layer": args.layer,
        "feature_mean": mean.cpu(),
        "feature_std": std.cpu(),
        "lambda_val": args.lambda_val,
        "test_metrics": test_metrics,
        "test_sample_idxs": [s["sample_idx"] for s in test_s],
        "val_sample_idxs": [s["sample_idx"] for s in val_s],
        "train_sample_idxs": [s["sample_idx"] for s in train_s],
    }, args.out_ckpt)
    print(f"\nsaved indicator to {args.out_ckpt}")


if __name__ == "__main__":
    main()