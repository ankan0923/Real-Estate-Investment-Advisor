"""
Command-line training script (model training + evaluation + MLflow tracking).

    python train.py                 # train on a 60,000-row sample, log to MLflow
    python train.py --rows 100000   # use a larger sample
    python train.py --no-mlflow     # skip MLflow logging
    python train.py --export        # also write the cleaned dataset to data/processed_housing.csv

The Streamlit app loads the saved models from ./models/model_bundle.joblib.
"""
import argparse
import time

import housing_pipeline as hp


def main():
    ap = argparse.ArgumentParser(description="Train the EstateIQ models")
    ap.add_argument("--rows", type=int, default=hp.DEFAULT_TRAIN_ROWS, help="training sample size")
    ap.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    ap.add_argument("--export", action="store_true", help="save the cleaned dataset as CSV")
    args = ap.parse_args()

    path = hp.find_dataset()
    if path is None:
        raise SystemExit("india_housing_prices.csv not found (put it in this folder or in ./data).")

    t0 = time.time()
    print(f"Loading and cleaning {path} ...")
    df, ref, report = hp.prepare_dataset(path)
    print(f"  {report['raw_rows']:,} raw rows -> {report['clean_rows']:,} clean rows "
          f"({report['duplicates_removed']:,} duplicates, {report['invalid_removed']:,} invalid removed)")
    print(f"  Good-investment share: {report['good_share']:.1f}%")

    if args.export:
        hp.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(hp.PROCESSED_PATH, index=False)
        print(f"  Cleaned dataset saved to {hp.PROCESSED_PATH}")

    def progress(msg, frac):
        print(f"  [{frac:4.0%}] {msg}")

    bundle = hp.train_all(df, ref, max_rows=args.rows, log_mlflow=not args.no_mlflow, progress=progress)
    path = hp.save_bundle(bundle)

    print("\nClassification (Good Investment)")
    for name, m in (bundle["clf"] or {"metrics": {}})["metrics"].items():
        star = " *" if name == bundle["clf"]["champion"] else ""
        print(f"  {name:<22}{star:<2} accuracy={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")
    print("Regression (price after 5 years)")
    for name, m in bundle["reg"]["metrics"].items():
        star = " *" if name == bundle["reg"]["champion"] else ""
        print(f"  {name:<22}{star:<2} rmse={m['rmse']:.3f} mae={m['mae']:.3f} r2={m['r2']:.4f}")

    ml = bundle["mlflow"]
    if ml["enabled"]:
        print(f"\nMLflow: {ml['runs']} runs logged. View them with:\n  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    elif ml["error"]:
        print(f"\nMLflow logging skipped: {ml['error']}")
    print(f"\nModels saved to {path}  (total {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
