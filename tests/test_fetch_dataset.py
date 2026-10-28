"""scripts/fetch_dataset.py: URL downloads that end short are retried; a file matching its pin is kept."""
import hashlib
import importlib.util
import io
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_dataset", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "fetch_dataset.py")
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)


class Response(io.BytesIO):
    def __init__(self, body: bytes, length: int):
        super().__init__(body)
        self.headers = {"Content-Length": str(length)}


@pytest.fixture
def server(monkeypatch):
    calls = []
    monkeypatch.setattr(fd.time, "sleep", lambda s: None)

    def serve(bodies):
        def urlopen(url, timeout):
            calls.append(url)
            return Response(bodies[min(len(calls), len(bodies)) - 1], 10)
        monkeypatch.setattr(fd.urllib.request, "urlopen", urlopen)
        return calls
    return serve


def test_a_short_transfer_is_retried(tmp_path, server):
    calls = server([b"01234", b"0123456789"])
    fd.fetch_url("https://x/f", tmp_path / "f.z01")
    assert (tmp_path / "f.z01").read_bytes() == b"0123456789" and len(calls) == 2
    assert not (tmp_path / "f.z01.part").exists()


def test_every_transfer_short_stops_the_fetch(tmp_path, server):
    server([b"012"])
    with pytest.raises(SystemExit):
        fd.fetch_url("https://x/f", tmp_path / "f", tries=2)
    assert not (tmp_path / "f").exists() and not (tmp_path / "f.part").exists()


def test_a_file_matching_its_pin_is_not_downloaded_again(tmp_path, server):
    calls = server([b"0123456789"])
    (tmp_path / "f").write_bytes(b"kept")
    fd.fetch_url("https://x/f", tmp_path / "f", hashlib.sha256(b"kept").hexdigest())
    assert calls == [] and (tmp_path / "f").read_bytes() == b"kept"
