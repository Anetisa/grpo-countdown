"""Plot the GRPO training curve from a JSONL metrics log.

Usage:
    python scripts/plot_training.py results_train.jsonl --out assets/training_curve.png
"""
import argparse, json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--out", default="assets/training_curve.png")
    a = p.parse_args()

    its, acc, reward, kl = [], [], [], []
    with open(a.log) as f:
        for line in f:
            r = json.loads(line)
            its.append(r["iter"]); acc.append(r["acc"])
            reward.append(r["reward"]); kl.append(r["kl"])

    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(its, acc, label="accuracy (reward=1)")
    ax1.plot(its, reward, label="mean reward", alpha=0.7)
    ax1.set_xlabel("iteration"); ax1.set_ylabel("value")
    ax1.set_title("Countdown accuracy / reward"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(its, kl, color="tab:red")
    ax2.set_xlabel("iteration"); ax2.set_ylabel("KL to reference")
    ax2.set_title("KL (should stay small)"); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
