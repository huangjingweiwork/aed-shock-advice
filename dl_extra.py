"""Download the extra PhysioNet databases for v2 into data/physionet/ (CinC 2015 training set and SDDB)."""
import time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW = Path(__file__).parent / "data" / "physionet"
BASE = "https://physionet.org/files"


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            break
        except urllib.error.HTTPError as e:  # some SDDB records have no .atr file
            print("skip", url, e.code, flush=True)
            return
        except OSError as e:  # timeouts and dropped connections: wait and retry
            print("retry", attempt + 1, url, e, flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError("failed: " + url)
    tmp.replace(dest)


def get(db, files, sub=""):
    out = RAW / db
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{BASE}/{db.replace('_', '/1.0.0/') if sub else db + '/1.0.0'}/{f}", out / f) for f in files]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))
    print("done", db, len(files), "files", flush=True)


if __name__ == "__main__":
    recs = urllib.request.urlopen(f"{BASE}/challenge-2015/1.0.0/training/RECORDS").read().decode().split()
    get("challenge-2015_training", [f"{r}.{e}" for r in recs for e in ("hea", "mat")], sub="training")
    recs = urllib.request.urlopen(f"{BASE}/sddb/1.0.0/RECORDS").read().decode().split()
    get("sddb", [f"{r}.{e}" for r in recs for e in ("hea", "dat", "atr", "ari")])
