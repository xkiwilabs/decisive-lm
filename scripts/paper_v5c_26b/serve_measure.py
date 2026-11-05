"""HTTP measurements of a DecisiveLM server: latency on a ~1,200-token, 20-option request, the longest requests
(text and text + photo) near DECISIVELM_MAX_TOKENS, and an audio request on a backbone without audio."""
import base64, io, json, math, statistics, struct, sys, time, urllib.error, urllib.request, wave

URL = sys.argv[1]            # e.g. http://127.0.0.1:8004
TEXT = open(sys.argv[2]).read()   # long English text to cut states from
PHOTO = sys.argv[3]


def post(body):
    req = urllib.request.Request(URL + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
            return r.status, out, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.perf_counter() - t0


words = TEXT.split()


def state(n_words, offset=0):
    return " ".join(words[offset:offset + n_words])


def choice_qs(n_q, n_opt, opt_words):
    return {f"q{j}": {"type": "choice", "instructions": f"Which of these best describes part {j} of the text? " + state(6, 50 * j),
                      "criteria": {f"o{i}": state(opt_words, 1000 + 97 * j + 13 * i) for i in range(n_opt)}}
            for j in range(n_q)}


# 1. Standard request: ~1,000-token state, 4 Choice questions of 5 options.
std = {"state": state(int(sys.argv[4])), "questions": choice_qs(4, 5, 6)}
code, out, _ = post(std)
print("standard:", code, out.get("usage"), flush=True)
times = []
for i in range(25):
    code, out, t = post(std)
    assert code == 200, out
    times.append(t)
times = sorted(times[5:])
print(f"standard HTTP: {out['usage']['input_tokens']} tokens, {out['usage']['blocks']} blocks: "
      f"p50 {statistics.median(times)*1000:.0f} ms, p95 {times[int(0.95*len(times))-1]*1000:.0f} ms; "
      f"server latency_ms {out['usage']['latency_ms']:.0f}", flush=True)

# 2. Longest text request: a state cut at 8,192 tokens plus about 4,000 tokens of questions.
for n_q in (14, 13, 12, 11):
    long = {"state": state(9000), "questions": choice_qs(n_q, 10, 12)}
    code, out, t = post(long)
    print(f"long text, {n_q} questions: {code}", out.get("usage") or out.get("error"), f"{t:.1f} s", flush=True)
    if code == 200:
        break

# 3. The same with a photo.
photo = {"type": "image", "data": base64.b64encode(open(PHOTO, "rb").read()).decode()}
for n_q in (n_q, n_q - 2, n_q - 4, n_q - 6):
    body = {"state": state(9000), "media": [photo], "questions": choice_qs(n_q, 10, 12)}
    code, out, t = post(body)
    print(f"long text + photo, {n_q} questions: {code}", out.get("usage") or out.get("error"), f"{t:.1f} s", flush=True)
    if code == 200:
        break

# 4. Audio: one second of a 440 Hz tone.
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * k / 16000))) for k in range(16000)))
code, out, _ = post({"state": "An audio recording.", "media": [{"type": "audio", "data": base64.b64encode(buf.getvalue()).decode()}],
                     "questions": {"speech": {"type": "noul", "instructions": "Someone is speaking in the recording"}}})
print("audio:", code, out, flush=True)
