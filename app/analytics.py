
from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT_DIR / "reports" / "generated"

FIGURE_DIR = REPORT_DIR / "figures"

ALLOWED_FIGURES = {
    "confusion_matrix.png",
    "roc_curve.png",
    "precision_recall_curve.png"
}


def read_csv_records(path):
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)

        df = df.where(
            pd.notna(df),
            None
        )

        return df.to_dict(orient="records")

    except Exception:
        return []


def get_model_comparisons():
    if not REPORT_DIR.exists():
        return []

    comparison_files = sorted(
        list(REPORT_DIR.glob("*model*comparison*.csv"))
        + list(REPORT_DIR.glob("*tuned*comparison*.csv"))
    )

    results = []

    seen = set()

    for path in comparison_files:
        if path.name in seen:
            continue

        seen.add(path.name)

        records = read_csv_records(path)

        if records:
            results.append({
                "file": path.name,
                "results": records
            })

    return results


def get_shap_importance():
    path = REPORT_DIR / "shap_feature_importance.csv"

    return read_csv_records(path)


def get_available_figures():
    figures = []

    for filename in sorted(ALLOWED_FIGURES):
        path = FIGURE_DIR / filename

        if path.is_file():
            figures.append({
                "name": filename,
                "url": f"/api/analytics/figure/{filename}"
            })

    return figures


def get_analytics():
    return {
        "model_comparisons": get_model_comparisons(),
        "shap_importance": get_shap_importance(),
        "figures": get_available_figures()
    }


def get_figure_path(filename):
    if filename not in ALLOWED_FIGURES:
        return None

    path = FIGURE_DIR / filename

    if not path.is_file():
        return None

    return path
