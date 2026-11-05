"""DecisiveLM over HTTP (docs/guide/04 §4.10): one backbone in memory, any number of bundles trained on it, chosen per
request by the request's `model` field.

    uv sync --extra serve
    DECISIVELM_BUNDLES=bundles/general_v3d_e4b_r8_seed0 uv run python -m decisivelm.serve

    GET  /health          200 when the model is loaded; no auth (for health probes)
    GET  /v1/models       the bundles this server holds
    POST /v1/systemone    a System One request (docs/guide/04); `model` names a bundle, or is left out for the default

Configured only by environment variables, so the same code runs as a local service or in a container:

    DECISIVELM_BUNDLES      a bundle directory, a directory of bundles, or several of either joined by commas (required)
    DECISIVELM_DEFAULT      the bundle used when a request names none (default: the first)
    DECISIVELM_HOST         address(es) to bind, joined by commas (default 127.0.0.1; never 0.0.0.0 outside a container)
    DECISIVELM_PORT         default 8003
    DECISIVELM_DEVICE       e.g. cuda:0 (default: cuda, else mps, else cpu)
    DECISIVELM_MEMORY_GB    a hard cap on this process's GPU memory (CUDA): an oversized request fails here, not in a
                            neighbour on the same GPU
    DECISIVELM_MAX_TOKENS   the longest packed request accepted (default 12288: a state cut at its 8,192-token budget
                            plus about 4,000 tokens of questions; about 18.6 GiB on E4B); longer ones get 413
    DECISIVELM_AUTH         none (default) | keys | litellm. With none, the server refuses to bind anything but
                            loopback unless DECISIVELM_ALLOW_UNAUTHENTICATED=1
    DECISIVELM_API_KEYS     Bearer tokens accepted when AUTH=keys, joined by commas
    DECISIVELM_LITELLM_URL  when AUTH=litellm, a token is accepted if this LiteLLM proxy accepts it (GET /v1/models);
                            answers are cached for 5 minutes (a refusal for 10 s), at most 10,000 at a time, so a
                            deleted or rotated key keeps working for up to 5 minutes

Media must come as base64 `data`: a server never reads a path from a request (docs/guide/04 §4.5). Requests are
answered one at a time on the GPU; a bundle switch copies its LoRA weights into the shared backbone (about 70 MB).
Each answered request is logged with the first 12 hex digits of its key's sha256 (never the key), the bundle, the
number of questions, tokens and milliseconds, so usage can be counted per key.
"""
import dataclasses
import hashlib
import hmac
import json
import logging
import os
import pathlib
import socket
import threading
import time
import urllib.error
import urllib.request

import torch

from decisivelm.api_types import SystemOneRequest
from decisivelm.bundle import attach_bundle, bundle_backbone, bundle_calibration
from decisivelm.head import Calibration, Scorer
from decisivelm.infer import DecisiveLM, RequestTooLarge
from decisivelm.lora import load_lora_state

LORA_SHAPE = ("rank", "alpha", "target_regex")
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
log = logging.getLogger("decisivelm.serve")


@dataclasses.dataclass
class Settings:
    bundles: list[pathlib.Path]
    default: str | None = None
    hosts: list[str] = dataclasses.field(default_factory=lambda: ["127.0.0.1"])
    port: int = 8003
    device: str | None = None
    memory_gb: float | None = None
    max_tokens: int = 12_288
    auth: str = "none"
    api_keys: list[str] = dataclasses.field(default_factory=list)
    litellm_url: str | None = None
    allow_unauthenticated: bool = False

    @classmethod
    def from_env(cls, env=os.environ) -> "Settings":
        def split(name):
            return [x.strip() for x in env.get(name, "").split(",") if x.strip()]
        if not split("DECISIVELM_BUNDLES"):
            raise SystemExit("DECISIVELM_BUNDLES is not set")
        s = cls(bundles=find_bundles(split("DECISIVELM_BUNDLES")), default=env.get("DECISIVELM_DEFAULT") or None,
                hosts=split("DECISIVELM_HOST") or ["127.0.0.1"], port=int(env.get("DECISIVELM_PORT", 8003)),
                device=env.get("DECISIVELM_DEVICE") or None,
                memory_gb=float(env["DECISIVELM_MEMORY_GB"]) if env.get("DECISIVELM_MEMORY_GB") else None,
                max_tokens=int(env.get("DECISIVELM_MAX_TOKENS", 12_288)), auth=env.get("DECISIVELM_AUTH", "none"),
                api_keys=split("DECISIVELM_API_KEYS"), litellm_url=env.get("DECISIVELM_LITELLM_URL") or None,
                allow_unauthenticated=env.get("DECISIVELM_ALLOW_UNAUTHENTICATED") == "1")
        if s.auth not in ("none", "keys", "litellm"):
            raise SystemExit(f"DECISIVELM_AUTH must be none, keys or litellm, not {s.auth!r}")
        if s.auth == "keys" and not s.api_keys:
            raise SystemExit("DECISIVELM_AUTH=keys needs DECISIVELM_API_KEYS")
        if s.auth == "litellm" and not s.litellm_url:
            raise SystemExit("DECISIVELM_AUTH=litellm needs DECISIVELM_LITELLM_URL")
        exposed = [h for h in s.hosts if h not in LOOPBACK]
        if s.auth == "none" and exposed and not s.allow_unauthenticated:
            raise SystemExit(f"binding {', '.join(exposed)} without auth: set DECISIVELM_AUTH, or "
                             "DECISIVELM_ALLOW_UNAUTHENTICATED=1 if only trusted callers can reach that address")
        return s


def find_bundles(entries: list[str]) -> list[pathlib.Path]:
    """Each entry is a bundle (a directory with calibration.json) or a directory of them, taken in name order."""
    out = []
    for e in map(pathlib.Path, entries):
        if (e / "calibration.json").exists():
            out.append(e)
        elif e.is_dir():
            out += sorted(p for p in e.iterdir() if (p / "calibration.json").exists())
        else:
            raise SystemExit(f"{e}: not a bundle or a directory of bundles")
    if len({p.name for p in out}) != len(out):
        raise SystemExit("two bundles share a name; each bundle is addressed by its directory name")
    if not out:
        raise SystemExit("DECISIVELM_BUNDLES holds no bundle")
    return out


@dataclasses.dataclass
class _Entry:
    adapter: dict[str, torch.Tensor]
    scorer: Scorer
    calibration: Calibration
    image_budget: int | None


class Bundles:
    """Several bundles on one backbone. All must be LoRA bundles of the same shape on the same backbone revision; the
    first is attached (which adds the LoRA modules) and the others' weights are copied in when they are used."""

    def __init__(self, paths: list[pathlib.Path], backbone, tokenizer, default: str | None = None, **media):
        paths = [pathlib.Path(p) for p in paths]
        calibs = {p.name: json.loads((p / "calibration.json").read_text()) for p in paths}
        first = calibs[paths[0].name]
        for p in paths:
            c = calibs[p.name]
            if not (p / "adapter.pt").exists():
                raise ValueError(f"{p.name}: no adapter.pt; the server takes LoRA bundles only")
            if c.get("backbone") != first.get("backbone"):
                raise ValueError(f"{p.name} was trained on a different backbone from {paths[0].name}")
            if [c["lora"][k] for k in LORA_SHAPE] != [first["lora"][k] for k in LORA_SHAPE]:
                raise ValueError(f"{p.name}'s LoRA shape differs from {paths[0].name}'s")
        self.dlm: DecisiveLM = attach_bundle(paths[0], backbone, tokenizer)
        for k, v in media.items():
            setattr(self.dlm, k, v)
        self.default_budget = self.dlm.image_budget
        device = next(backbone.parameters()).device
        self.entries = {}
        for p in paths:
            scorer = Scorer(backbone.config.get_text_config().hidden_size)
            scorer.load_state_dict(torch.load(p / "scorer.pt", weights_only=True, map_location="cpu"))
            self.entries[p.name] = _Entry(torch.load(p / "adapter.pt", weights_only=True, map_location="cpu"),
                                          scorer.to(device).eval(), bundle_calibration(calibs[p.name]),
                                          calibs[p.name].get("image_budget"))
        self.default = default or paths[0].name
        if self.default not in self.entries:
            raise ValueError(f"default bundle {self.default!r} is not among {sorted(self.entries)}")
        self.current = paths[0].name
        self.lock = threading.Lock()

    def use(self, name: str | None) -> DecisiveLM:
        """The model with bundle `name` in place (the default for None). Call with self.lock held."""
        name = name or self.default
        e = self.entries[name]
        if name != self.current:
            load_lora_state(self.dlm.backbone, e.adapter)
            self.current = name
        self.dlm.scorer, self.dlm.calibration, self.dlm.adapter = e.scorer, e.calibration, name
        self.dlm.image_budget = e.image_budget or self.default_budget
        return self.dlm


def load(s: Settings) -> Bundles:
    """The backbone the bundles were trained on, its media processors, and the bundles."""
    from decisivelm.model import backbone_profile, load_backbone, load_feature_extractor, load_image_processor
    name = bundle_backbone(s.bundles[0])
    dtype = getattr(torch, json.loads((s.bundles[0] / "calibration.json").read_text()).get("dtype", "bfloat16"))
    if s.memory_gb and torch.cuda.is_available():
        dev = torch.device(s.device or "cuda")
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, s.memory_gb * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory), dev)
    backbone, tokenizer = load_backbone(name, dtype=dtype, device=s.device)
    profile, media = backbone_profile(name), {}
    if "image" in profile.get("modalities", []):
        media |= {"image_processor": load_image_processor(name), "image_budget": profile["image_max_soft_tokens"]}
    if "audio" in profile.get("modalities", []):
        media["feature_extractor"] = load_feature_extractor(name)
    return Bundles(s.bundles, backbone, tokenizer, default=s.default, **media)


def litellm_checker(url: str, ttl: float = 300.0, refusal_ttl: float = 10.0, max_entries: int = 10_000):
    """A function token -> bool that asks a LiteLLM proxy whether it accepts the token. Only digests are cached,
    at most `max_entries`: when full, expired answers go first, then the oldest."""
    cache, lock = {}, threading.Lock()

    def check(token: str) -> bool:
        key, now = hashlib.sha256(token.encode()).hexdigest(), time.monotonic()
        with lock:
            hit = cache.get(key)
        if hit and hit[1] > now:
            return hit[0]
        req = urllib.request.Request(url.rstrip("/") + "/v1/models", headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                ok = r.status == 200
        except urllib.error.HTTPError as e:
            if e.code >= 500:
                raise ConnectionError(f"the key service answered {e.code}") from e
            ok = False
        with lock:
            if len(cache) >= max_entries:
                for k in [k for k, (_, until) in cache.items() if until <= now] or [next(iter(cache))]:
                    del cache[k]
            cache[key] = (ok, now + (ttl if ok else refusal_ttl))
        return ok
    return check


def create_app(bundles: Bundles, s: Settings, check_token=None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="DecisiveLM", docs_url=None, redoc_url=None, openapi_url=None)
    if s.auth == "litellm" and check_token is None:
        check_token = litellm_checker(s.litellm_url)

    def error(status: int, message: str, kind: str):
        return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)

    def key_id(request: Request) -> str:
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header[:7].lower() == "bearer " else ""
        return hashlib.sha256(token.encode()).hexdigest()[:12] if token else "-"

    def refused(request: Request):
        if s.auth == "none":
            return None
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header[:7].lower() == "bearer " else ""
        if not token:
            return error(401, "missing Bearer token", "authentication_error")
        if s.auth == "keys":
            ok = any(hmac.compare_digest(token, k) for k in s.api_keys)
        else:
            try:
                ok = check_token(token)
            except (OSError, ConnectionError):
                return error(503, "the key service is unavailable", "service_unavailable")
        return None if ok else error(401, "invalid Bearer token", "authentication_error")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models(request: Request):
        if r := refused(request):
            return r
        return {"object": "list", "default": bundles.default,
                "data": [{"id": n, "object": "model", "owned_by": "decisivelm"} for n in bundles.entries]}

    @app.post("/v1/systemone")
    def systemone(request: Request, body: dict):
        if r := refused(request):
            return r
        try:
            req = SystemOneRequest.model_validate(body)
        except ValueError as e:
            return error(400, str(e), "invalid_request_error")
        if any(m.path is not None for m in req.media):
            return error(400, "media must be base64 data; a server never reads paths", "invalid_request_error")
        if req.model is not None and req.model not in bundles.entries:
            return error(404, f"no bundle {req.model!r}; GET /v1/models lists them", "not_found_error")
        with bundles.lock:
            dlm = bundles.use(req.model)
            try:
                out = dlm.system_one(req, max_tokens=s.max_tokens)
            except RequestTooLarge as e:
                return error(413, str(e), "request_too_large")
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                return error(503, "out of GPU memory for this request; send fewer questions or a shorter state",
                             "service_unavailable")
            except ValueError as e:
                return error(400, str(e), "invalid_request_error")
            warnings = list(dlm.warnings)
        out["model"] = req.model or bundles.default
        log.info("key=%s model=%s questions=%d tokens=%d ms=%.0f", key_id(request), out["model"], len(req.questions),
                 out["usage"]["input_tokens"], out["usage"]["latency_ms"])
        headers = {"X-DecisiveLM-Warnings": json.dumps(warnings)} if warnings else None
        return JSONResponse(out, headers=headers)

    return app


def main():
    import uvicorn
    s = Settings.from_env()
    bundles = load(s)
    sockets = []
    for host in s.hosts:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, s.port))
        sockets.append(sock)
    print(f"DecisiveLM: {len(bundles.entries)} bundle(s), default {bundles.default}, on "
          + ", ".join(f"{h}:{s.port}" for h in s.hosts), flush=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.Server(uvicorn.Config(create_app(bundles, s), log_level="info")).run(sockets=sockets)


if __name__ == "__main__":
    main()
