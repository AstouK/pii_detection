"""
evaluate.py
───────────
Evaluation module for the two-stage GDPR PII detection pipeline.

Metrics are computed at three levels:
  1. Document-level  → detected_pii / llm_pii / final_pii vs contains_personal_data
  2. Sweep 1 strong vs any PII
  3. Per-entity-type → e.g. EMAIL_ADDRESS_yes_no vs per_type_conf detection

Ground-truth columns follow the naming convention:  <ENTITY_TYPE>_yes_no
"""

import pandas as pd

# ─────────────────────────────────────────────────────────────
# Entity types to evaluate at per-type level
# Add / remove entries to match your labeled columns
# ─────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "PASSPORT",
    "MEDICAL_LICENSE",
]

_TRUTHY = {"yes", "true", "1", "y", "ja"}


# ─────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convert yes/no (or True/False / 1/0) Series to boolean. NaN → False."""
    return series.fillna(False).astype(str).str.strip().str.lower().isin(_TRUTHY)


def normalise_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all ground-truth columns to booleans in-place.
    Must be called before any metric computation.
    """
    gt_cols = ["contains_personal_data"] + [f"{e}_yes_no" for e in ENTITY_TYPES]
    for col in gt_cols:
        if col in df.columns:
            df[col] = _to_bool_series(df[col])
    return df


# ─────────────────────────────────────────────────────────────
# Entity-type detection helper
# ─────────────────────────────────────────────────────────────


def _entity_detected(row: pd.Series, entity_type: str) -> bool:
    """Return True if entity_type appears in Sweep 1's per_type_conf dict."""
    conf_dict = row.get("per_type_conf", {})
    if not isinstance(conf_dict, dict):
        return False
    return entity_type in conf_dict


# ─────────────────────────────────────────────────────────────
# Core metric helpers
# ─────────────────────────────────────────────────────────────


def confusion(y_true: pd.Series, y_pred: pd.Series):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = (y_pred & y_true).sum()
    tn = (~y_pred & ~y_true).sum()
    fp = (y_pred & ~y_true).sum()
    fn = (~y_pred & y_true).sum()
    return int(tp), int(tn), int(fp), int(fn)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, label: str = "") -> dict:
    tp, tn, fp, fn = confusion(y_true, y_pred)
    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "label": label,
        "n": n,
        "accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


# ─────────────────────────────────────────────────────────────
# Document-level metrics (pred column already in df)
# ─────────────────────────────────────────────────────────────


def _doc_metrics(df: pd.DataFrame, pred_col: str, label: str) -> dict:
    if pred_col not in df.columns:
        return {}
    return compute_metrics(df["ground_truth_pii"], df[pred_col].astype(bool), label)


# ─────────────────────────────────────────────────────────────
# Per-entity-type metrics
# ─────────────────────────────────────────────────────────────


def compute_entity_metrics(df: pd.DataFrame) -> dict[str, dict]:
    """
    For each entity type that has a ground-truth column in the data,
    compute metrics against Sweep 1's per_type_conf detections.

    Returns dict keyed by entity type.
    """
    results = {}
    for etype in ENTITY_TYPES:
        gt_col = f"{etype}_yes_no"
        if gt_col not in df.columns:
            continue
        gt_bool = df[gt_col].astype(bool)
        pred_bool = df.apply(lambda r: _entity_detected(r, etype), axis=1)
        results[etype] = compute_metrics(gt_bool, pred_bool, label=f"sweep1_{etype}")
    return results


# ─────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────


def _print_block(m: dict) -> None:
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1 Score : {m['f1']:.4f}")
    print(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  n={m['n']}")


def _print_entity_table(entity_metrics: dict[str, dict]) -> None:
    if not entity_metrics:
        print("  (no per-type ground-truth columns found)")
        return

    header = f"  {'Entity Type':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}  {'n':>4}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for etype, m in entity_metrics.items():
        print(
            f"  {etype:<20} "
            f"{m['accuracy']:>6.4f} "
            f"{m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} "
            f"{m['f1']:>6.4f}  "
            f"{m['TP']:>4} {m['FP']:>4} {m['TN']:>4} {m['FN']:>4}  {m['n']:>4}"
        )


# ─────────────────────────────────────────────────────────────
# Main print_metrics (replaces the original)
# ─────────────────────────────────────────────────────────────


def print_metrics(
    df: pd.DataFrame,
    provider: str | None = None,
) -> dict:
    """
    Compute and print evaluation metrics for:
      - Sweep 1 (detected_pii  — strong PII only)
      - Sweep 1 any (detected_any_pii — strong + potential)
      - Sweep 2 (llm_pii, if present)
      - Final decision (final_pii, if present)
      - Per-entity-type Sweep 1 accuracy

    Expects df to have 'ground_truth_pii' already set (boolean).
    Call normalise_ground_truth(df) before this function.

    Returns dict of all metric dicts for downstream use.
    """

    provider_label = provider or "unknown"

    sep = "=" * 50

    print(f"\n{sep}")
    print(f"Provider: {provider}")
    print(f"{sep}\n")

    metrics = {}

    # ── Sweep 1: strong PII ───────────────────────────────────
    print("▶ Sweep 1 — Strong PII  (detected_pii)")
    m1 = _doc_metrics(df, "detected_pii", "sweep1_strong")
    if m1:
        _print_block(m1)
        metrics["sweep1_strong"] = m1

    # ── Sweep 1: any PII ─────────────────────────────────────
    print("\n▶ Sweep 1 — Any PII  (detected_any_pii)")
    m1a = _doc_metrics(df, "detected_any_pii", "sweep1_any")
    if m1a:
        _print_block(m1a)
        metrics["sweep1_any"] = m1a

    # ── Sweep 2 ───────────────────────────────────────────────
    if "llm_pii" in df.columns:
        print("\n▶ Sweep 2 — LLM Review  (llm_pii, flagged rows only)")
        flagged = df[df["needs_llm_review"] == True].copy() if "needs_llm_review" in df.columns else df
        m2 = compute_metrics(
            flagged["ground_truth_pii"],
            flagged["llm_pii"].astype(bool),
            label=f"sweep2_{provider_label}",
        )
        _print_block(m2)
        metrics["sweep2_llm"] = m2

    # ── Final decision ────────────────────────────────────────
    if "final_pii" in df.columns:
        print("\n▶ Final Decision  (final_pii = Sweep 1 OR Sweep 2)")
        m3 = _doc_metrics(
            df,
            "final_pii",
            f"final_{provider_label}",
        )
        if m3:
            _print_block(m3)
            metrics["final"] = m3

        if m1 and m3:
            print(f"\n  Accuracy gain Sweep 1 → Final : {m3['accuracy'] - m1['accuracy']:+.4f}")
            print(f"  F1 gain       Sweep 1 → Final : {m3['f1'] - m1['f1']:+.4f}")

    # ── Per-entity-type ───────────────────────────────────────
    print(f"\n{sep}")
    print("  PER-ENTITY-TYPE METRICS  (Sweep 1 detections)")
    print(f"{sep}\n")
    entity_metrics = compute_entity_metrics(df)
    _print_entity_table(entity_metrics)
    metrics["entity"] = entity_metrics

    print(f"\n{sep}\n")
    return metrics


# ─────────────────────────────────────────────────────────────
# Summary DataFrame (for saving / further analysis)
# ─────────────────────────────────────────────────────────────


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    """
    Flatten the metrics dict returned by print_metrics into a tidy DataFrame.
    One row per stage / entity type.
    """
    rows = []
    for key, val in metrics.items():
        if key == "entity":
            for etype, m in val.items():
                rows.append(m)
        elif isinstance(val, dict) and "accuracy" in val:
            rows.append(val)
    return pd.DataFrame(rows).set_index("label") if rows else pd.DataFrame()
