"""Download official xv6 at a recorded commit without using Git.

First invocation resolves the upstream riscv branch. Later invocations reuse
vendor/xv6-source.json unless --commit explicitly selects a revision.
"""
import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "OT-LAB8-local-setup"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", help="Full upstream SHA; omit to reuse pin or resolve riscv")
    args = parser.parse_args()
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    pin = vendor / "xv6-source.json"
    target = vendor / "xv6-riscv-baseline"
    if target.exists():
        raise SystemExit("Baseline directory exists; refusing to overwrite recorded source.")
    previous = json.loads(pin.read_text()) if pin.exists() else {}
    revision = args.commit or previous.get("commit") or "riscv"
    api_url = f"https://api.github.com/repos/mit-pdos/xv6-riscv/commits/{revision}"
    commit_data = json.loads(download(api_url))
    sha = commit_data["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Unexpected upstream commit identifier")
    archive_url = f"https://codeload.github.com/mit-pdos/xv6-riscv/tar.gz/{sha}"
    archive_bytes = download(archive_url)
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    expected = previous.get("archive_sha256")
    if expected and previous.get("commit") == sha and expected != archive_sha:
        raise ValueError("Archive hash differs from recorded download; inspect before proceeding")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        prefix = f"xv6-riscv-{sha}"
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != prefix:
                raise ValueError("Unsafe archive member")
            if not (member.isfile() or member.isdir()):
                raise ValueError("Unexpected non-file archive member")
        archive.extractall(vendor, filter="data")
    (vendor / prefix).rename(target)
    files = {
        str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(target.rglob("*")) if path.is_file()
    }
    license_text = (target / "LICENSE").read_text()
    metadata = {
        "repository": "https://github.com/mit-pdos/xv6-riscv",
        "commit": sha,
        "commit_url": f"https://github.com/mit-pdos/xv6-riscv/commit/{sha}",
        "commit_date": commit_data["commit"]["committer"]["date"],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_url": archive_url,
        "archive_sha256": archive_sha,
        "source_directory": str(target.relative_to(ROOT)),
        "license_file": "vendor/xv6-riscv-baseline/LICENSE",
        "license_note": "MIT-style license; authoritative text retained in LICENSE",
        "license_sha256": hashlib.sha256(license_text.encode()).hexdigest(),
        "acquisition": "GitHub API and immutable source archive; no Git commands",
    }
    pin.write_text(json.dumps(metadata, indent=2) + "\n")
    (vendor / "xv6-source-manifest.json").write_text(json.dumps(files, indent=2) + "\n")
    (vendor / f"xv6-{sha}.tar.gz").write_bytes(archive_bytes)
    print(json.dumps(metadata, indent=2))
    print(f"Recorded {len(files)} original source hashes.")


if __name__ == "__main__":
    main()
