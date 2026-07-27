import pandas as pd
from pii_detector import run_presidio_regex
from llm_reviewer import run_llm
from evaluate_detector import normalise_ground_truth, print_metrics, metrics_to_dataframe
from pii_detector import run_presidio_regex, _has_person_hint


# ── Load labeled test data ────────────────────────────────────
df = pd.read_excel(
    r"classification\data\sample_data.xlsx",
    parse_dates=["file_created_date", "last_modified_date"],
)

# ── Normalise all ground-truth columns (yes/no → bool) ───────
df = normalise_ground_truth(df)  # handles contains_personal_data + all <TYPE>_yes_no cols

# ── Rename document-level ground truth for metric functions ──
df = df.rename(columns={"contains_personal_data": "ground_truth_pii"})
assert "ground_truth_pii" in df.columns, \
    "Evaluation requires a 'ground_truth_pii' column."

# ── Sweep 1: Presidio + Regex ─────────────────────────────────
df = run_presidio_regex(df)

# ── Debug: inspect what goes to LLM ──────────────────────────
flagged = df[df["needs_llm_review"]]
print(f"\n[pipeline] Rows flagged for LLM: {len(flagged)} / {len(df)}")
print(f"[pipeline] full_text null count: {df['full_text'].isna().sum()}")
print(f"[pipeline] full_text empty string count: {(df['full_text'] == '').sum()}")

if not flagged.empty:
    sample = flagged.iloc[0]
    print(f"\n[pipeline] Sample flagged row:")
    print(f"  full_text length : {len(str(sample.get('full_text', '') or ''))}")
    print(f"  full_text preview: {repr(str(sample.get('full_text', ''))[:200])}")
    print(f"  entities         : {sample.get('entities', 'MISSING')}")
    print(f"  detected_pii     : {sample.get('detected_pii')}")
    print(f"  potential_cats   : {sample.get('potential_pii_categories')}")
else:
    print("\n[pipeline] ⚠ No rows flagged — check needs_llm_review logic")
    print(f"  detected_pii True count    : {df['detected_pii'].sum()}")
    print(f"  potential_pii nonempty     : {df['potential_pii_categories'].apply(lambda x: len(x) > 0).sum()}")
    print(f"  person_hint True count     : {df['full_text'].apply(_has_person_hint).sum()}")


# ── Sweep 2: LLM review (flagged rows only) ───────────────────
df = run_llm(df)

# ── Final decision ────────────────────────────────────────────
df["final_pii"] = df["detected_pii"] | df["llm_pii"]

# ── Evaluate & print ──────────────────────────────────────────
metrics = print_metrics(df)

# ── Optional: save full results to Excel ─────────────────────
#summary = metrics_to_dataframe(metrics)
#summary.to_excel(r"classification\data\eval_results.xlsx")
