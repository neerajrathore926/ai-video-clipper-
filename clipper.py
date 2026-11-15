"""AI clipper: video -> transcript -> best moments -> 9:16 clips with captions.

Pipeline: yt-dlp (download) -> faster-whisper (word-level transcript, GPU)
-> Ollama local LLM (pick highlights, heuristic fallback) -> ffmpeg (cut, crop, captions).
"""
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"      # any model you have pulled in Ollama
WHISPER_MODEL = "large-v3"       # use "medium" or "small" if you run low on VRAM
MIN_LEN, MAX_LEN = 20, 60        # clip length limits in seconds
CHUNK_SECONDS = 600              # transcript is sent to the LLM in 10-minute chunks


def log(cb, msg):
    print(msg, flush=True)
    if cb:
        cb(msg)


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ---------------------------------------------------------------- download
def download(url, workdir, cb=None):
    log(cb, "Downloading video with yt-dlp...")
    template = str(Path(workdir) / "source.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist",
           "-f", "bv*[height<=1080]+ba/b[height<=1080]",
           "--merge-output-format", "mp4", "-o", template]
    browser = os.environ.get("CLIPPER_COOKIES_BROWSER")  # e.g. "chrome" or "safari"
    if browser:
        cmd += ["--cookies-from-browser", browser]
    r = subprocess.run(cmd + [url], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "yt-dlp could not download the video. Its error was:\n"
            + (r.stderr or r.stdout)[-1200:]
            + "\n\nCommon fixes: 1) pip install -U yt-dlp  2) install ffmpeg (brew install ffmpeg)  "
              "3) install a JavaScript runtime (brew install deno)  "
              "4) if it says 'confirm you're not a bot', start the app with "
              "CLIPPER_COOKIES_BROWSER=chrome python app.py  5) or download the video yourself and upload the file."
        )
    return Path(workdir) / "source.mp4"


# -------------------------------------------------------------- transcribe
def _run_whisper(video, model_name, device, compute_type):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    seg_iter, _info = model.transcribe(str(video), word_timestamps=True, vad_filter=True)
    segs, words = [], []
    for s in seg_iter:
        segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        for w in s.words or []:
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
    return segs, words


def transcribe(video, cb=None):
    log(cb, "Transcribing (this is the slow step)...")
    try:
        return _run_whisper(video, WHISPER_MODEL, "cuda", "float16")
    except Exception as e:  # missing CUDA libs, out of memory, etc.
        log(cb, f"GPU transcription failed ({e}); falling back to CPU with 'small' model.")
        return _run_whisper(video, "small", "cpu", "int8")


# --------------------------------------------------------------- highlights
def _ask_llm(chunk_segs, per_chunk, cb):
    lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in chunk_segs)
    prompt = (
        "You choose the best moments of a video transcript for YouTube Shorts.\n"
        f"Each clip must be {MIN_LEN}-{MAX_LEN} seconds, start at the beginning of a sentence, "
        "end when a thought is complete, open with a strong hook and make sense on its own.\n"
        f"Pick up to {per_chunk} clips. Give each a score from 1 to 10 and a short title.\n"
        'Reply ONLY with JSON: {"clips":[{"start":0.0,"end":0.0,"title":"...","score":8}]}\n'
        "Transcript lines look like [start-end] text, in seconds.\n\n" + lines
    )
    body = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"num_ctx": 16384, "temperature": 0.3},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return json.loads(data["response"]).get("clips", [])


def _heuristic(segs, n):
    """Fallback without an LLM: score ~35s windows by speech density and emphasis."""
    hooks = re.compile(r"\b(secret|never|always|mistake|why|how|best|worst|truth|actually|important)\b", re.I)
    cands = []
    for i, s in enumerate(segs):
        j, text = i, ""
        while j < len(segs) and segs[j]["end"] - s["start"] < 35:
            text += " " + segs[j]["text"]
            j += 1
        if j >= len(segs):
            break
        dur = segs[j]["end"] - s["start"]
        if dur > MAX_LEN:
            continue
        score = len(text.split()) / dur + 0.5 * (text.count("?") + text.count("!")) + 0.4 * len(hooks.findall(text))
        cands.append({"start": s["start"], "end": segs[j]["end"], "title": text.strip()[:50], "score": score})
    return cands


def _non_overlapping(clips, n):
    chosen = []
    for c in sorted(clips, key=lambda c: -float(c.get("score", 0))):
        if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return sorted(chosen, key=lambda c: c["start"])


def pick_highlights(segs, n, cb=None):
    log(cb, "Finding highlights...")
    clips = []
    try:
        total = segs[-1]["end"]
        t = 0.0
        while t < total:
            chunk = [s for s in segs if t <= s["start"] < t + CHUNK_SECONDS]
            if chunk:
                clips += _ask_llm(chunk, 3, cb)
            t += CHUNK_SECONDS
        log(cb, f"LLM suggested {len(clips)} candidates.")
    except Exception as e:
        log(cb, f"Local LLM unavailable ({e}); using simple heuristic instead.")
        clips = _heuristic(segs, n)

    starts = [s["start"] for s in segs]
    ends = [s["end"] for s in segs]
    cleaned = []
    for c in clips:
        try:
            st = min(starts, key=lambda x: abs(x - float(c["start"])))
            en = min(ends, key=lambda x: abs(x - float(c["end"])))
        except (KeyError, TypeError, ValueError):
            continue
        if en - st > MAX_LEN:
            en = st + MAX_LEN
        if en - st < MIN_LEN:
            continue
        cleaned.append({"start": st, "end": en, "title": str(c.get("title", "clip")),
                        "score": c.get("score", 0)})
    if not cleaned:
        cleaned = _heuristic(segs, n)
    return _non_overlapping(cleaned, n)


# ----------------------------------------------------------------- captions
def _ass_time(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,84,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
WHITE, YELLOW = r"{\c&H00FFFFFF&}", r"{\c&H0000FFFF&}"


def write_ass(words, clip_start, clip_end, path, group=3):
    ws = [w for w in words if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.3]
    events = []
    for g in range(0, len(ws), group):
        grp = ws[g:g + group]
        texts = [w["word"].upper().replace("{", "").replace("}", "") for w in grp]
        for i, w in enumerate(grp):
            start = w["start"] - clip_start
            end = (grp[i + 1]["start"] if i + 1 < len(grp) else w["end"]) - clip_start
            if end <= start:
                end = start + 0.1
            line = " ".join((YELLOW + t + WHITE) if k == i else t for k, t in enumerate(texts))
            events.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{WHITE}{line}")
    Path(path).write_text(ASS_HEADER + "\n".join(events) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- render
def render(src, start, end, ass_name, out_name, workdir, mode="crop"):
    """mode 'crop' = fill the vertical frame (center crop); 'blur' = full video over blurred background."""
    if mode == "blur":
        vf = ["-filter_complex",
              "[0:v]split=2[a][b];"
              "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
              "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,"
              f"ass={ass_name}[v]",
              "-map", "[v]", "-map", "0:a?"]
    else:
        vf = ["-vf", f"crop='trunc(ih*9/16/2)*2':ih,scale=1080:1920,setsar=1,ass={ass_name}"]

    base = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(Path(src).resolve()),
            "-t", f"{end - start:.2f}", *vf]
    tail = ["-c:a", "aac", "-b:a", "160k", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_name]
    gpu = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", "-b:v", "0"]
    cpu = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]

    # cwd = workdir so the ass filter gets a plain relative filename (avoids path-escaping issues)
    r = subprocess.run(base + gpu + tail, cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run(base + cpu + tail, cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-1500:])


# ----------------------------------------------------------------- pipeline
def run_pipeline(source, workdir, n_clips=5, mode="crop", cb=None):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    video = download(source, workdir, cb) if str(source).startswith("http") else Path(source)

    segs, words = transcribe(video, cb)
    if not segs:
        raise RuntimeError("No speech found in the video.")
    clips = pick_highlights(segs, n_clips, cb)

    results = []
    for i, c in enumerate(clips, 1):
        log(cb, f"Rendering clip {i}/{len(clips)}: {c['title']} ({mmss(c['start'])}-{mmss(c['end'])})")
        ass, out = f"clip{i}.ass", f"clip{i}.mp4"
        write_ass(words, c["start"], c["end"], workdir / ass)
        render(video, c["start"], c["end"], ass, out, workdir, mode)
        results.append({"file": out, "title": c["title"], "start": c["start"], "end": c["end"]})
    (workdir / "clips.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(cb, "Done.")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Make 9:16 captioned clips from a video URL or file.")
    ap.add_argument("source", help="YouTube URL or path to a video file")
    ap.add_argument("-n", type=int, default=5, help="number of clips")
    ap.add_argument("--mode", choices=["crop", "blur"], default="crop")
    ap.add_argument("-o", "--out", default="output")
    a = ap.parse_args()
    run_pipeline(a.source, a.out, a.n, a.mode)
