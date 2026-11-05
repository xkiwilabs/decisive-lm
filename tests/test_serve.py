"""decisivelm.serve: bundles switched on one backbone answer as each bundle alone; auth, limits and errors."""
import json

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from decisivelm import serve  # noqa: E402
from decisivelm.bundle import attach_bundle  # noqa: E402
from decisivelm.head import Scorer, bucket  # noqa: E402
from decisivelm.infer import QUERY_IDS  # noqa: E402
from decisivelm.lora import apply_lora, lora_state  # noqa: E402
from decisivelm.model import backbone_profile  # noqa: E402
from decisivelm.preprocess import MAX_STATE_TOKENS  # noqa: E402
from tiny_gemma import tiny_decoder, tokenizer  # noqa: E402

REGEX = backbone_profile("e4b")["lora_target_regex"]
REQ = {"state": "Our API started returning 500s 20 minutes ago.",
       "questions": {"d": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Payment issues", "technical": "Bugs or outages",
                                        "sales": "Pricing"}},
                     "u": {"type": "noul", "instructions": "The message is urgent"}}}


def _bundle(path, seed, rank=4):
    torch.manual_seed(seed)
    m = tiny_decoder("dense", "sdpa", vocab=len(tokenizer()))
    apply_lora(m, REGEX, rank=rank, alpha=8, query_ids=list(QUERY_IDS.values()))
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith(("lora_B", "query_rows.delta")):
                p.normal_(0.0, 0.05)
    path.mkdir(parents=True)
    torch.save(lora_state(m), path / "adapter.pt")
    torch.save(Scorer(m.config.hidden_size).state_dict(), path / "scorer.pt")
    (path / "calibration.json").write_text(json.dumps(
        {"temperatures": {f"choice:{bucket(3)}": 1.0 + seed / 10, "noul": 1.2}, "b_noul": 0.1 * seed,
         "lora": {"rank": rank, "alpha": 8, "dropout": 0.05, "target_regex": REGEX}}))
    return path


@pytest.fixture
def bundles(tmp_path):
    return [_bundle(tmp_path / "b" / "alpha", 1), _bundle(tmp_path / "b" / "beta", 2)]


def _client(bundles, check=None, **settings):
    torch.manual_seed(0)
    b = serve.Bundles(bundles, tiny_decoder("dense", "sdpa", vocab=len(tokenizer())), tokenizer())
    return TestClient(serve.create_app(b, serve.Settings(bundles=bundles, **settings), check_token=check))


def _alone(path):
    torch.manual_seed(0)
    return attach_bundle(path, tiny_decoder("dense", "sdpa", vocab=len(tokenizer())), tokenizer()).system_one(REQ)


def test_each_bundle_answers_as_it_does_alone_whatever_was_used_before(bundles):
    c = _client(bundles)
    expected = {p.name: _alone(p)["answers"] for p in bundles}
    for name in ["beta", "alpha", None, "beta"]:
        r = c.post("/v1/systemone", json=REQ | ({"model": name} if name else {}))
        assert r.status_code == 200
        got, want = r.json()["answers"], expected[name or "alpha"]
        assert max(abs(got["d"]["probabilities"][k] - want["d"]["probabilities"][k]) for k in want["d"]["probabilities"]) < 1e-6
        assert abs(got["u"]["noul"] - want["u"]["noul"]) < 1e-6
        assert r.json()["model"] == (name or "alpha") and r.json()["usage"]["adapter"] == (name or "alpha")
    assert expected["alpha"]["u"]["noul"] != expected["beta"]["u"]["noul"]


def test_health_needs_no_key_and_everything_else_does(bundles):
    c = _client(bundles, auth="keys", api_keys=["k1"])
    assert c.get("/health").status_code == 200
    assert c.get("/v1/models").status_code == 401
    assert c.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer nope"}).status_code == 401
    r = c.get("/v1/models", headers={"Authorization": "Bearer k1"})
    assert r.status_code == 200 and [m["id"] for m in r.json()["data"]] == ["alpha", "beta"]


def test_usage_is_logged_per_key_digest_never_the_key(bundles, caplog):
    c = _client(bundles, auth="keys", api_keys=["secret-key-1"])
    with caplog.at_level("INFO", logger="decisivelm.serve"):
        assert c.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer secret-key-1"}).status_code == 200
    line = next(r.getMessage() for r in caplog.records if r.name == "decisivelm.serve")
    assert "secret-key-1" not in line and "key=" in line and "model=alpha questions=2" in line


def test_litellm_auth_asks_the_checker_and_reports_it_down(bundles):
    calls = []

    def check(token):
        calls.append(token)
        if token == "down":
            raise ConnectionError("no")
        return token == "good"
    c = _client(bundles, auth="litellm", litellm_url="http://x", check=check)
    assert c.get("/v1/models", headers={"Authorization": "Bearer good"}).status_code == 200
    assert c.get("/v1/models", headers={"Authorization": "Bearer bad"}).status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer down"}).status_code == 503
    assert calls == ["good", "bad", "down"]


def test_paths_unknown_bundles_bad_requests_and_oversized_requests_are_refused(bundles):
    c = _client(bundles, max_tokens=40)
    small = {"state": "hi", "questions": {"u": {"type": "noul", "instructions": "urgent"}}}
    assert c.post("/v1/systemone", json=small).status_code == 200
    assert c.post("/v1/systemone", json=small | {"model": "gamma"}).status_code == 404
    assert c.post("/v1/systemone", json=small | {"media": [{"type": "image", "path": "/etc/passwd"}]}).status_code == 400
    assert c.post("/v1/systemone", json={"state": "hi", "questions": {}}).status_code == 400
    r = c.post("/v1/systemone", json=REQ)
    assert r.status_code == 413 and "tokens" in r.json()["error"]["message"]


def test_bundles_of_another_lora_shape_are_refused(tmp_path):
    a, b = _bundle(tmp_path / "a", 1), _bundle(tmp_path / "b", 2, rank=8)
    with pytest.raises(ValueError, match="LoRA shape"):
        serve.Bundles([a, b], tiny_decoder("dense", "sdpa", vocab=len(tokenizer())), tokenizer())


def test_settings_come_from_the_environment(bundles):
    root = bundles[0].parent
    s = serve.Settings.from_env({"DECISIVELM_BUNDLES": str(root), "DECISIVELM_HOST": "127.0.0.1, 172.17.0.1",
                                 "DECISIVELM_MEMORY_GB": "20", "DECISIVELM_DEFAULT": "beta",
                                 "DECISIVELM_AUTH": "litellm", "DECISIVELM_LITELLM_URL": "http://127.0.0.1:4200"})
    assert [p.name for p in s.bundles] == ["alpha", "beta"] and s.hosts == ["127.0.0.1", "172.17.0.1"]
    assert s.memory_gb == 20 and s.port == 8003 and s.default == "beta" and s.auth == "litellm"
    # a state at its truncation budget must still leave room for its questions under the default limit
    assert s.max_tokens >= MAX_STATE_TOKENS + 4_000
    with pytest.raises(SystemExit, match="API_KEYS"):
        serve.Settings.from_env({"DECISIVELM_BUNDLES": str(root), "DECISIVELM_AUTH": "keys"})
    with pytest.raises(SystemExit, match="not set"):
        serve.Settings.from_env({})


def test_an_exposed_bind_needs_auth_or_an_explicit_opt_out(bundles):
    env = {"DECISIVELM_BUNDLES": str(bundles[0].parent), "DECISIVELM_HOST": "0.0.0.0"}
    with pytest.raises(SystemExit, match="without auth"):
        serve.Settings.from_env(env)
    assert serve.Settings.from_env(env | {"DECISIVELM_AUTH": "keys", "DECISIVELM_API_KEYS": "k"}).auth == "keys"
    assert serve.Settings.from_env(env | {"DECISIVELM_ALLOW_UNAUTHENTICATED": "1"}).allow_unauthenticated
    assert serve.Settings.from_env(env | {"DECISIVELM_HOST": "127.0.0.1,::1"}).auth == "none"


def test_litellm_checker_caches_digests_only(monkeypatch):
    seen = []

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(req, timeout):
        seen.append(req.headers["Authorization"])
        return Resp()
    monkeypatch.setattr(serve.urllib.request, "urlopen", fake)
    check = serve.litellm_checker("http://hub:4000/", max_entries=3)
    assert check("sk-1") and check("sk-1") and seen == ["Bearer sk-1"]
    cache = next(c.cell_contents for c in check.__closure__ if isinstance(c.cell_contents, dict))
    assert "sk-1" not in repr(cache)
    for t in ["sk-2", "sk-3", "sk-4", "sk-5"]:
        check(t)
    assert len(cache) <= 3
