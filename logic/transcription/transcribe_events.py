from pathlib import Path
from pydub import AudioSegment
import requests, uuid, time
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.assemblyai.com/v2"
PRE_PAD = POST_PAD = 8  # in seconds

def clip_and_transcribe_events(events, wav_path, api_key, max_workers=5):
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
        cs = max(0, ev["start"] - PRE_PAD)
        ce = ev["end"] + POST_PAD
        clip = audio[int(cs * 1000):int(ce * 1000)]

        clip_dir = Path("clips")
        clip_dir.mkdir(exist_ok=True)
        tmp_path = clip_dir / f"{uuid.uuid4().hex}.wav"
        clip.export(tmp_path, format="wav")

        up = requests.post(f"{API}/upload", headers=headers_upload, data=open(tmp_path, "rb")).json()
        tid = requests.post(f"{API}/transcript", headers=headers_transcribe,
                            json={"audio_url": up["upload_url"], "punctuate": True}).json()["id"]
        return idx, tid, cs, ev

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(clip_upload, (i, ev)) for i, ev in enumerate(events)]
        for fu in as_completed(futures):
            idx, tid, cs, ev = fu.result()
            clip_results[tid] = dict(idx=idx, clip_start=cs, ev=ev)
            pending.add(tid)

    # Poll AssemblyAI
    while pending:
        time.sleep(4)
        for tid in list(pending):
            js = requests.get(f"{API}/transcript/{tid}", headers=headers_upload).json()
            if js["status"] in ("completed", "error"):
                clip_results[tid]["words"] = js.get("words", [])
                pending.remove(tid)
        print(f"⌛ Waiting… {len(pending)} jobs remaining")

    return clip_results
