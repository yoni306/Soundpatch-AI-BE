from pathlib import Path
from pydub import AudioSegment
import requests, uuid, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

API = "https://api.assemblyai.com/v2"
PRE_PAD = POST_PAD = 10  # in seconds

def clip_and_transcribe_events(events: List[Dict[str, Any]], wav_path: str, api_key: str, max_workers: int = 5) -> Dict[str, Any]:
    headers_upload = {"authorization": api_key}
    headers_transcribe = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    audio = AudioSegment.from_file(wav_path)
    clip_results = {}
    pending = set()

    def clip_upload(idx_event):
        idx, ev = idx_event
        # Use the correct field names from our noise detection model
        event_start = ev["start_time"]
        event_end = ev["end_time"]
        
        # Calculate clip start and end with padding, but don't go below 0 or beyond audio length
        audio_duration = len(audio) / 1000.0  # Convert to seconds
        cs = max(0, event_start - PRE_PAD)
        ce = min(audio_duration, event_end + POST_PAD)
        
        clip = audio[int(cs * 1000):int(ce * 1000)]

        clip_dir = Path("clips")
        clip_dir.mkdir(exist_ok=True)
        tmp_path = clip_dir / f"{uuid.uuid4().hex}.wav"
        clip.export(tmp_path, format="wav")

        try:
            # Upload audio file
            with open(tmp_path, "rb") as f:
                up = requests.post(f"{API}/upload", headers=headers_upload, data=f).json()
            
            # Request transcription
            tid = requests.post(f"{API}/transcript", headers=headers_transcribe,
                                json={"audio_url": up["upload_url"], "punctuate": True}).json()["id"]
            
            # Clean up temporary file
            os.remove(tmp_path)
            
            return idx, tid, cs, ev
            
        except Exception as e:
            # Clean up temporary file on error
            if tmp_path.exists():
                os.remove(tmp_path)
            print(f"Error processing clip {idx}: {e}")
            return idx, None, cs, ev
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(clip_upload, (i, ev)) for i, ev in enumerate(events)]
        for fu in as_completed(futures):
            idx, tid, cs, ev = fu.result()
            # Skip if transcription failed
            if tid is not None:
                clip_results[tid] = dict(idx=idx, clip_start=cs, ev=ev)
                pending.add(tid)

    # Poll AssemblyAI
    while pending:
        time.sleep(4)
        for tid in list(pending):
            try:
                js = requests.get(f"{API}/transcript/{tid}", headers=headers_upload).json()
                if js["status"] in ("completed", "error"):
                    clip_results[tid]["words"] = js.get("words", [])
                    if js["status"] == "error":
                        print(f"⚠️ Transcription error for {tid}: {js.get('error', 'Unknown error')}")
                    pending.remove(tid)
            except Exception as e:
                print(f"⚠️ Error polling transcription {tid}: {e}")
                pending.remove(tid)
        
        if pending:
            print(f"⌛ Waiting… {len(pending)} jobs remaining")

    return clip_results

