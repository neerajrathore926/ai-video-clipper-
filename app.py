"""Local web UI for the clipper. Run: python app.py  then open http://127.0.0.1:5000"""
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import clipper

app = Flask(__name__)
JOBS_DIR = Path("jobs")
JOBS = {}

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Clipper</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;background:#111;color:#eee}
  input,select,button{font-size:16px;padding:10px;border-radius:8px;border:1px solid #444;background:#222;color:#eee}
  input[type=text]{width:100%;box-sizing:border-box}
  button{background:#e5322d;border:0;cursor:pointer}
  button:disabled{opacity:.5}
  .row{display:flex;gap:12px;margin:12px 0;flex-wrap:wrap;align-items:center}
  pre{background:#1b1b1b;padding:12px;border-radius:8px;max-height:200px;overflow:auto;white-space:pre-wrap}
  .clips{display:flex;gap:16px;flex-wrap:wrap}
  .clip{width:240px}
  video{width:240px;border-radius:10px;background:#000}
  a{color:#7ab8ff}
</style></head><body>
<h2>AI Clipper</h2>
<p>Paste a YouTube URL or upload a video. You get vertical 9:16 clips with captions.</p>
<input id="url" type="text" placeholder="https://www.youtube.com/watch?v=...">
<div class="row"><span>or upload:</span><input id="file" type="file" accept="video/*,audio/*"></div>
<div class="row">
  <label>Clips <input id="n" type="number" value="5" min="1" max="15" style="width:70px"></label>
  <label>Framing
    <select id="mode"><option value="crop">Fill (center crop)</option><option value="blur">Fit (blurred background)</option></select>
  </label>
  <button id="go">Make clips</button>
</div>
<pre id="log">Waiting...</pre>
<div class="clips" id="clips"></div>
<script>
const $ = id => document.getElementById(id);
$("go").onclick = async () => {
  const fd = new FormData();
  fd.append("url", $("url").value.trim());
  fd.append("n", $("n").value);
  fd.append("mode", $("mode").value);
  if ($("file").files[0]) fd.append("file", $("file").files[0]);
  $("go").disabled = true; $("clips").innerHTML = ""; $("log").textContent = "Starting...";
  const r = await fetch("/start", {method: "POST", body: fd});
  const j = await r.json();
  if (j.error) { $("log").textContent = j.error; $("go").disabled = false; return; }
  poll(j.id);
};
async function poll(id) {
  const j = await (await fetch("/status/" + id)).json();
  $("log").textContent = j.log.join("\\n");
  $("log").scrollTop = $("log").scrollHeight;
  if (j.status === "running") { setTimeout(() => poll(id), 2000); return; }
  $("go").disabled = false;
  if (j.status === "error") { $("log").textContent += "\\nERROR: " + j.error; return; }
  $("clips").innerHTML = j.results.map(c =>
    `<div class="clip"><video controls src="/files/${id}/${c.file}"></video>
     <div>${c.title}</div><a href="/files/${id}/${c.file}" download>Download</a></div>`).join("");
}
</script></body></html>"""


def _worker(job_id, source, n, mode):
    job = JOBS[job_id]
    try:
        job["results"] = clipper.run_pipeline(source, JOBS_DIR / job_id, n, mode, cb=job["log"].append)
        job["status"] = "done"
    except Exception as e:
        job["status"], job["error"] = "error", str(e)


@app.get("/")
def index():
    return PAGE


@app.post("/start")
def start():
    url = request.form.get("url", "").strip()
    f = request.files.get("file")
    n = max(1, min(int(request.form.get("n", 5)), 15))
    mode = request.form.get("mode", "crop")
    if not url and not f:
        return jsonify(error="Paste a URL or upload a file.")
    job_id = uuid.uuid4().hex[:8]
    (JOBS_DIR / job_id).mkdir(parents=True, exist_ok=True)
    if f and f.filename:
        source = JOBS_DIR / job_id / ("upload_" + secure_filename(f.filename))
        f.save(source)
        source = str(source)
    else:
        source = url
    JOBS[job_id] = {"status": "running", "log": [], "results": [], "error": ""}
    threading.Thread(target=_worker, args=(job_id, source, n, mode), daemon=True).start()
    return jsonify(id=job_id)


@app.get("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    return jsonify(job) if job else (jsonify(error="unknown job"), 404)


@app.get("/files/<job_id>/<name>")
def files(job_id, name):
    return send_from_directory((JOBS_DIR / job_id).resolve(), name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
