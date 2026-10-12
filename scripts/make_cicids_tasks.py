"""
Make the "unseen" evaluation tasks: real CICIDS2017 rows from the TEST split
(the models never trained on them), sampled per attack label, run through the
real Detection Manager.

Expected outcomes:
    benign rows  -> no_action
    attack rows  -> act, and the technique acted on must fit the row's true
                    category (escalating a real attack counts as over-cautious,
                    which is safe but not a success)

Needs the dataset (kagglehub, Kaggle credentials) and the trained models for
running the tasks afterwards (python -m src.detection.train).

Usage (from the repo root):
    python scripts/make_cicids_tasks.py                      # 2 rows per attack label + 8 benign
    python scripts/make_cicids_tasks.py --per-label 3 --benign 10 --seed 7
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.training import data  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "evaluation" / "tasks" / "unseen_tasks.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-label", type=int, default=2, help="rows per attack label")
    parser.add_argument("--benign", type=int, default=8, help="benign rows")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    df = data.load_clean_dataframe()
    _, test = data.split_train_test(df)  # the shared split: these rows were never trained on
    features = data.select_feature_names(test)
    test = test.assign(_fine=test[data.LABEL_COL].map(data.map_cicids_label_to_fine))

    tasks = []
    for fine, group in sorted(test.groupby("_fine"), key=lambda kv: kv[0]):
        is_benign = fine == data.BENIGN_LABEL
        n = args.benign if is_benign else args.per_label
        rows = group.sample(n=min(n, len(group)), random_state=args.seed)
        category = data.FINE_LABEL_TO_CATEGORY.get(fine)
        for k, (_, row) in enumerate(rows.iterrows(), 1):
            task = {
                "id": f"U-{fine.replace(' ', '')}-{k}",
                "difficulty": "simple" if is_benign else "moderate",
                "split": "unseen",
                "description": f"Real CICIDS2017 test-split row, true label {fine}",
                "features": {f: float(row[f]) for f in features},
                "mitigation": "real",
                "expected": ["no_action"] if is_benign else ["act"],
            }
            if category:
                task["true_category"] = category
            tasks.append(task)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
    print(f"Wrote {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()