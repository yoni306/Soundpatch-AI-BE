def summarize_transcriptions(clip_results):
    result = []
    for data in clip_results.values():
        ev = data["ev"]
        offset_ms = data["clip_start"] * 1000
        ns, ne = ev["start"] * 1000, ev["end"] * 1000
        words = data.get("words", [])

        before = [w["text"] for w in words if (w["end"] + offset_ms) <= ns]
        after  = [w["text"] for w in words if (w["start"] + offset_ms) >= ne]
        inside = [w["text"] for w in words if ns <= (w["start"] + offset_ms) <= ne]

        result.append({
            "clip_id": data["idx"],
            **ev,
            "start_time": round(ev["start"], 2),
            "end_time": round(ev["end"], 2),
            "missing_text": " ".join(inside),
            "context_before": " ".join(before[-50:]),
            "context_after": " ".join(after[:50])
        })
    return result
