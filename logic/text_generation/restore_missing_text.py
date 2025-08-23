import os, json, math
import pandas as pd
# import google.generativeai as genai  # Commented out due to protobuf conflicts
from typing import List, Dict, Any

# ================== GEMINI via REST (no SDK, no protobuf) ==================
import time, requests

GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # חינמי ומהיר

def gemini_rest_generate(
    prompt: str,
    api_key: str = None,
    model: str = GEMINI_DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 120,
    retries: int = 3,
    timeout: int = 45
) -> str:
    """
    מחזיר טקסט מה-Gemini ללא SDK. לא דורש protobuf ולא google-generativeai.
    """
    api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        return "[Gemini error: missing GEMINI_API_KEY]"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=body, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                # חילוץ בטוח של הטקסט מהתשובה
                try:
                    cands = data.get("candidates", [])
                    if not cands:
                        return "[No response]"
                    parts = cands[0].get("content", {}).get("parts", [])
                    texts = []
                    for p in parts:
                        t = p.get("text")
                        if t:
                            texts.append(t)
                    return " ".join(texts).strip() if texts else "[No response]"
                except Exception as e:
                    return f"[Gemini parse error: {e}]"
            # קודי שגיאה נפוצים: 429/403/5xx
            if r.status_code in (429, 500, 502, 503, 504):
                wait_time = 5 * attempt  # חכה יותר זמן במקרה של rate limit
                print(f"⏳ Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            return f"[Gemini HTTP {r.status_code}: {r.text[:200]}]"
        except Exception as e:
            last_err = e
            time.sleep(1.2 * attempt)
    return f"[Gemini network error after retries: {last_err}]"

# ------------------ SETUP --------------------------------

def configure_gemini(api_key: str):
    # genai.configure(api_key=api_key)
    # return genai  # Return the module itself for compatibility
    print("⚠️ Using REST API instead of SDK due to protobuf conflicts")
    return None  # We'll use REST API instead


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

def generate_prompt_text(context_before: str, partial_inside: str, context_after: str, duration: float) -> str:
    return (
        "You are a professional audio transcription specialist. Your ONLY task is to predict the exact missing words.\n\n"
        
        f"SITUATION: A {duration:.1f}-second audio segment was corrupted during recording.\n"
        f"NOTE: In normal speech, people typically speak at 2-3 words per second, but you should decide based on context and flow.\n\n"
        
        "CONTEXT PROVIDED:\n"
        f"🔹 Words BEFORE the gap: \"{context_before}\"\n"
        f"🔹 Partial words DURING the gap: \"{partial_inside}\"\n" 
        f"🔹 Words AFTER the gap: \"{context_after}\"\n"
        f"🔹 Gap duration: {duration:.1f} seconds\n\n"
        
        "INSTRUCTIONS:\n"
        "1. Analyze the natural flow of speech from before → during → after\n"
        "2. Consider the {duration:.1f}-second duration as a guide for how much content was likely spoken\n"
        "3. Predict ONLY the missing words that would naturally bridge the gap\n"
        "4. Consider the speaker's pace, tone, grammar, and context\n"
        "5. Use any partial words as clues for what was being said\n"
        "6. The duration helps estimate length, but prioritize natural speech flow\n"
        "7. Keep it natural and conversational\n\n"
        
        "OUTPUT FORMAT: Return ONLY the missing words/phrase - no explanations, no quotes, no commentary.\n"
        "EXAMPLE: If missing words are 'and then we went', just return: and then we went"
    )


def restore_missing_text(
    gemini_model,
    clip_results: Dict[str, Any],
    log_path: str = None
) -> List[Dict[str, Any]]:
    """
    Restore missing text using Gemini based on transcription results.
    
    Args:
        gemini_model: Configured Gemini model
        clip_results: Results from clip_and_transcribe_events
        log_path: Optional path to save logs
        
    Returns:
        List of restored text results
    """
    log_blocks = []
    restored_results = []

    for i, (tid, data) in enumerate(clip_results.items()):
        # Extract transcribed words
        words = data.get("words", [])
        clip_start = data["clip_start"]
        event = data["ev"]
        
        # Calculate event timing relative to clip
        event_start_in_clip = event["start_time"] - clip_start
        event_end_in_clip = event["end_time"] - clip_start
        
        print(f"🔧 Debug clip {i+1}:")
        print(f"   📁 Clip start: {clip_start:.2f}s")
        print(f"   🔇 Event: {event['start_time']:.2f}s - {event['end_time']:.2f}s")
        print(f"   📐 Event in clip: {event_start_in_clip:.2f}s - {event_end_in_clip:.2f}s")
        print(f"   📝 Total words: {len(words)}")
        
        # Separate words into before, during, and after the noise event
        words_before = []
        words_during = []
        words_after = []
        
        for word in words:
            word_start = word.get("start", 0) / 1000.0  # Convert ms to seconds
            word_end = word.get("end", 0) / 1000.0
            word_text = word.get("text", "")
            
            if word_end <= event_start_in_clip:
                words_before.append(word_text)
            elif word_start >= event_end_in_clip:
                words_after.append(word_text)
            else:
                words_during.append(word_text)  # Partial words during noise
        
        print(f"   📊 Words distribution: Before={len(words_before)}, During={len(words_during)}, After={len(words_after)}")
        
        # Create context strings
        context_before = " ".join(words_before[-10:])  # Last 10 words before
        partial_inside = " ".join(words_during)  # Any partial words
        context_after = " ".join(words_after[:10])   # First 10 words after
        
        duration = event["end_time"] - event["start_time"]
        prompt = generate_prompt_text(context_before, partial_inside, context_after, duration)

        try:
            response = gemini_model.generate_text(
                prompt=prompt,
                temperature=0.3,
                max_output_tokens=100
            )
            restored_text = response.result.strip() if response.result else "[No response]"
        except Exception as e:
            restored_text = f"[Gemini error: {e}]"

        result = {
            "clip_id": i,
            "start_time": event["start_time"],
            "end_time": event["end_time"], 
            "duration": duration,
            "noise_type": event["noise_type"],
            "confidence": event["confidence"],
            "context_before": context_before,
            "partial_inside": partial_inside,
            "context_after": context_after,
            "restored_text": restored_text
        }

        log_block = (
            f"{'=' * 80}\n"
            f"Clip {i}  ({event['start_time']:.2f}-{event['end_time']:.2f}s)  "
            f"type={event['noise_type']} confidence={event['confidence']:.3f}\n\n"
            f"📝 Context before: \"{context_before}\"\n"
            f"🔇 Partial during: \"{partial_inside}\"\n"
            f"📝 Context after: \"{context_after}\"\n\n"
            f"🤖 Gemini prediction: \"{restored_text}\"\n"
        )
        log_blocks.append(log_block)
        restored_results.append(result)

    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_blocks)
        print(f"✅ Gemini log saved → {log_path} (total {len(restored_results)} clips)")

    return restored_results

# ------------------ PROMPT BUILDER (UPDATED) ------------------
def generate_prompt_text_improved(context_before: str, partial_inside: str, context_after: str, duration: float) -> str:
    return (
        "You are a professional audio transcription specialist. Your ONLY task is to predict the exact missing words.\n\n"
        f"SITUATION: A {duration:.1f}-second audio segment was corrupted during recording.\n"
        "NOTE: In normal speech, people typically speak at ~2–3 words per second; use context and natural flow.\n\n"
        "CONTEXT PROVIDED:\n"
        f"🔹 Words BEFORE the gap: \"{context_before}\"\n"
        f"🔹 Partial words DURING the gap: \"{partial_inside}\"\n"
        f"🔹 Words AFTER the gap: \"{context_after}\"\n"
        f"🔹 Gap duration: {duration:.1f} seconds\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the natural flow from before → during → after.\n"
        f"2. Consider the {duration:.1f}-second duration as a guide to the likely length.\n"
        "3. Predict ONLY the missing words that naturally bridge the gap.\n"
        "4. Use partial words as strong clues.\n"
        "5. Return ONLY the missing words/phrase — no quotes, no explanations."
    )


# ------------------ MAIN RESTORE (drop-in replacement) ------------------
def restore_missing_text_via_rest(
    clip_results: Dict[str, Any],
    api_key: str = None,
    log_path: str = None,
    temperature: float = 0.3,
    max_output_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    אותו API כמו הפונקציה שלך – רק שהקריאה ל-Gemini נעשית ב-REST.
    Expects clip_results[tid] to contain:
        - "words": list of {text,start,end} (ms)
        - "clip_start": float (sec)
        - "ev": {start_time,end_time,noise_type,confidence,filename,...} (sec)
    """
    log_blocks, out = [], []

    for i, (tid, data) in enumerate(clip_results.items()):
        words = data.get("words", []) or []
        clip_start = float(data["clip_start"])
        ev = data["ev"]

        ev_s, ev_e = float(ev["start_time"]), float(ev["end_time"])
        ev_s_in, ev_e_in = ev_s - clip_start, ev_e - clip_start

        before, during, after = [], [], []
        for w in words:
            ws = (w.get("start", 0) or 0) / 1000.0
            we = (w.get("end",   0) or 0) / 1000.0
            txt = (w.get("text") or "").strip()
            if not txt:
                continue
            if we <= ev_s_in:
                before.append(txt)
            elif ws >= ev_e_in:
                after.append(txt)
            else:
                during.append(txt)

        context_before = " ".join(before[-10:])
        partial_inside = " ".join(during)
        context_after  = " ".join(after[:10])
        duration = ev_e - ev_s

        prompt = generate_prompt_text_improved(context_before, partial_inside, context_after, duration)
        restored_text = gemini_rest_generate(
            prompt,
            api_key=api_key,
            model=GEMINI_DEFAULT_MODEL,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )

        out.append({
            "clip_id": i,
            "filename": ev.get("filename"),
            "start_time": ev_s,
            "end_time": ev_e,
            "duration": duration,
            "noise_type": ev.get("noise_type"),
            "confidence": float(ev.get("confidence", 0.0)),
            "context_before": context_before,
            "partial_inside": partial_inside,
            "context_after": context_after,
            "restored_text": restored_text
        })

        log_blocks.append(
            f"{'='*80}\n"
            f"Clip {i}  ({ev_s:.2f}-{ev_e:.2f}s)  type={ev.get('noise_type')}  conf={float(ev.get('confidence',0.0)):.3f}\n\n"
            f"📝 Before: \"{context_before}\"\n"
            f"🔇 During: \"{partial_inside}\"\n"
            f"📝 After:  \"{context_after}\"\n\n"
            f"🤖 Gemini: \"{restored_text}\"\n"
        )

    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_blocks)
        print(f"✅ Gemini log saved → {log_path} (total {len(out)} clips)")

    return out
