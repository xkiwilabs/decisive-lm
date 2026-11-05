"""Download a registered dataset's pinned files into data/raw/<name>/ (configs/datasets.yaml `fetch`).

    uv run python scripts/fetch_dataset.py banking77 clinc150

Each entry is either {url, file} (a URL pinned to a commit), {hf, revision, path, file} (a Hub dataset file at a
fixed revision), {hf, revision, allow} (every Hub file matching the glob `allow`, kept at its own path) or
{github, revision, allow} (every file of a GitHub repository at a commit matching `allow`, read through
media.githubusercontent.com so Git LFS files arrive as their content). An optional `sha256` pins the content too
(needed for a URL that is not pinned to a commit); a file that does not match is removed and the fetch stops. Writes
MANIFEST.json with each file's source and sha256.
"""
import argparse, concurrent.futures, fnmatch, hashlib, json, pathlib, shutil, sys, time, urllib.request

from huggingface_hub import hf_hub_download, snapshot_download

from decisivelm.data.dataset import load_registry
from decisivelm.model import hub_kwargs

REPO = pathlib.Path(__file__).resolve().parent.parent


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_url(url: str, dest: pathlib.Path, want: str | None = None, tries: int = 4):
    """Download to dest via a .part file, retrying a transfer that fails or ends short of its Content-Length. A file
    already here matching the pinned sha256 is kept."""
    if want and dest.exists() and sha256(dest) == want:
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as w:
                size = r.headers.get("Content-Length")
                shutil.copyfileobj(r, w, 1 << 20)
            if size is not None and part.stat().st_size != int(size):
                raise OSError(f"{part.stat().st_size:,} of {int(size):,} bytes")
            part.rename(dest)
            return
        except OSError as e:   # urllib.error.URLError and timeouts are OSErrors
            print(f"{dest.name}: attempt {attempt} failed ({e})", flush=True)
            time.sleep(10 * attempt)
    part.unlink(missing_ok=True)
    sys.exit(f"{dest.name}: no complete download after {tries} attempts")


def github_files(repo: str, revision: str, allow: str) -> list[str]:
    """Paths of the blobs at `revision` matching `allow`; the revision must be a full commit sha."""
    if len(revision) != 40:
        raise SystemExit(f"{repo}: pin a full commit sha, not {revision!r}")
    url = f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"
    with urllib.request.urlopen(url) as r:
        tree = json.load(r)
    if tree.get("truncated"):
        raise SystemExit(f"{repo}: the tree listing is truncated")
    return sorted(t["path"] for t in tree["tree"] if t["type"] == "blob" and fnmatch.fnmatchcase(t["path"], allow))


def fetch_github(f: dict, out: pathlib.Path) -> list[dict]:
    def one(rel):
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            url = f"https://media.githubusercontent.com/media/{f['github']}/{f['revision']}/{rel}"
            with urllib.request.urlopen(url) as r, open(dest.with_suffix(dest.suffix + ".part"), "wb") as w:
                shutil.copyfileobj(r, w)
            dest.with_suffix(dest.suffix + ".part").rename(dest)
        return {"file": rel, "github": f["github"], "revision": f["revision"], "path": rel,
                "bytes": dest.stat().st_size, "sha256": sha256(dest)}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, github_files(f["github"], f["revision"], f["allow"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--jobs", type=int, default=1,
                    help="URL files downloaded at once (a slow host such as Zenodo can be faster over several)")
    a = ap.parse_args()
    registry = load_registry()
    for name in a.names:
        entry = registry[name]
        out = REPO / "data" / "raw" / name
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"dataset": name, "license": entry["license"], "tier": entry["tier"], "files": []}
        fetched = set()
        if a.jobs > 1:
            urls = [f for f in entry["fetch"] if "url" in f]
            with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
                list(pool.map(lambda f: fetch_url(f["url"], out / f["file"], f.get("sha256")), urls))
            fetched = {f["file"] for f in urls}
        for f in entry["fetch"]:
            if "github" in f:   # a pinned commit's matching files (resumable: files already here are kept)
                files = fetch_github(f, out)
                manifest["files"] += files
                print(f"{name}: {len(files)} files matching {f['allow']}, {sum(x['bytes'] for x in files) / 1e6:.0f} MB")
                continue
            if "allow" in f:   # a pinned revision's matching files, each recorded like a single file
                snap = pathlib.Path(snapshot_download(f["hf"], repo_type="dataset", revision=f["revision"],
                                                      allow_patterns=f["allow"], **hub_kwargs(offline=False)))
                files = sorted(x for x in snap.rglob("*") if x.is_file())
                for x in files:
                    rel = x.relative_to(snap).as_posix()
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(x, out / rel)
                    manifest["files"].append({"file": rel, "hf": f["hf"], "revision": f["revision"], "path": rel,
                                              "bytes": x.stat().st_size, "sha256": sha256(out / rel)})
                print(f"{name}: {len(files)} files matching {f['allow']}")
                continue
            dest = out / f["file"]
            if "url" in f:
                if f["file"] not in fetched:
                    fetch_url(f["url"], dest, f.get("sha256"))
                source = {"url": f["url"]}
            else:
                shutil.copy(hf_hub_download(f["hf"], f["path"], repo_type="dataset", revision=f["revision"],
                                            **hub_kwargs(offline=False)), dest)
                source = {"hf": f["hf"], "revision": f["revision"], "path": f["path"]}
            digest = sha256(dest)
            if f.get("sha256", digest) != digest:   # a URL not pinned to a commit pins its content instead
                dest.unlink()
                sys.exit(f"{name}/{f['file']}: sha256 {digest} is not the pinned {f['sha256']}; removed")
            manifest["files"].append({"file": f["file"], **source, "bytes": dest.stat().st_size, "sha256": digest})
            print(f"{name}/{f['file']}: {dest.stat().st_size / 1e6:.1f} MB")
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
