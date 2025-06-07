import os, json, math
import pandas as pd
import google.generativeai as genai

# ------------------ SETUP --------------------------------

def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash-latest")


# ------------------ MATCHING GROUND TRUTH ------------------

def best_matching_gt_row(pred_row: dict, gt_df: pd.DataFrame) -> pd.Series:
    """Find GT row with ≥50% overlap, or nearest midpoint if not found."""
    gtf = gt_df[gt_df.filename == pred_row["filename"]].copy()
    gtf["inter"] = gtf.apply(lambda row:
        max(0, min(row.end_time, pred_row["end_time"]) -
                max(row.start_time, pred_row["start_time"])), axis=1)
    gtf["dur"] = gtf.end_time - gtf.start_time
    gtf["ratio"] = gtf["inter"] / gtf["dur"]

    candidates = gtf[gtf["ratio"] >= 0.5]
    if not candidates.empty:
        return candidates.sort_values("ratio", ascending=False).iloc[0]

    # If no sufficient overlap, use nearest center
    pred_mid = (pred_row["start_time"] + pred_row["end_time"]) / 2
    gtf["center"] = (gtf.start_time + gtf.end_time) / 2
    return gtf.iloc[(gtf.center - pred_mid).abs().argmin()]


# ------------------ GEMINI RESTORE -------------------------

def generate_prompt_text(r: dict, est_words: int) -> str:
    return (
        "You are an expert transcription restorer.\n"
        f"The missing segment lasted about {r['end_time'] - r['start_time']:.1f} seconds "
        f"(≈{est_words} words). Reconstruct exactly what was spoken, using:\n"
        "- context before\n- any partial words inside\n- context after\n\n"
        f"Context before:\n{r['context_before']}\n\n"
        f"Partial inside:\n{r['missing_text']}\n\n"
        f"Context after:\n{r['context_after']}\n\n"
        "Return only the sentence – no commentary."
    )


def restore_missing_text(
    gemini_model,
    pred_rows: list[dict],
    gt_df: pd.DataFrame,
    log_path: str = None
) -> list[dict]:
    log_blocks = []
    restored_results = []

    for i, r in enumerate(pred_rows):
        r["clip_id"] = i
        gt_row = best_matching_gt_row(r, gt_df)
        r["ground_truth_text"] = gt_row["missing_text"]

        seg_duration = r["end_time"] - r["start_time"]
        approx_words = max(1, math.ceil(seg_duration * 2.3))
        prompt = generate_prompt_text(r, approx_words)

        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 60, "temperature": 0.4}
            )
            r["restored_text"] = response.text.strip()
        except Exception as e:
            r["restored_text"] = f"[Gemini error: {e}]"

        log_block = (
            f"{'=' * 80}\n"
            f"Clip {r['clip_id']}  ({r['start_time']:.2f}-{r['end_time']:.2f}s)  "
            f"type={r['label'] if 'label' in r else r['noise_type']}\n\n"
            f"✅ Ground-truth text:\n{r['ground_truth_text']}\n\n"
            f"🤖 Gemini prediction:\n{r['restored_text']}\n"
        )
        log_blocks.append(log_block)
        restored_results.append(r)

    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_blocks)
        print(f"✅ Gemini log saved → {log_path} (total {len(restored_results)} clips)")

    return restored_results
