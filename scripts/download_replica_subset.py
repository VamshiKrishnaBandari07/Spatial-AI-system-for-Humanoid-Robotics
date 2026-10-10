"""Stream one official Replica geometry subset, caching a bounded archive prefix."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request


class ReleaseStream:
    def __init__(self, cache, limit):
        self.cache = Path(cache)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.saved = self.cache.stat().st_size if self.cache.exists() else 0
        self.replay = self.cache.open("rb") if self.saved else None
        self.writer = self.cache.open("ab")
        self.position = 0
        self.limit = limit
        self.response = None
        self.next_progress = 256 * 1024**2
        self.network_bytes = 0

    def read(self, size=-1):
        size = 10240 if size < 0 else size
        if self.position >= self.limit:
            raise ValueError("Replica prefix transfer budget exhausted")
        size = min(size, self.limit - self.position)
        if self.position < self.saved:
            data = self.replay.read(min(size, self.saved - self.position))
            self.position += len(data)
            return data
        part = self.position // 2000000000
        offset = self.position % 2000000000
        if part >= 17:
            raise ValueError("Official release stream ended")
        if self.response is None:
            url = f"https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta{chr(97 + part)}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
            self.response = urllib.request.urlopen(request, timeout=60)
            if self.response.status != 206 or not self.response.headers.get(
                "Content-Range", ""
            ).startswith(f"bytes {offset}-"):
                self.response.close()
                self.response = None
                raise ValueError("Official source must honor archive byte range")
            print(
                "Reading official release part",
                part + 1,
                "at offset",
                offset,
                flush=True,
            )
        data = self.response.read(min(size, 2000000000 - offset))
        if not data:
            self.response.close()
            self.response = None
            if offset:
                raise ValueError("Truncated official release part")
            return self.read(size)
        self.writer.write(data)
        self.position += len(data)
        self.network_bytes += len(data)
        if self.position >= self.next_progress:
            self.writer.flush()
            print(
                "Compressed prefix read MiB", round(self.position / 1024**2), flush=True
            )
            self.next_progress += 256 * 1024**2
        if self.position % 2000000000 == 0:
            self.response.close()
            self.response = None
        return data

    def close(self):
        if self.response:
            self.response.close()
        if self.replay:
            self.replay.close()
        self.writer.close()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="apartment_0")
    parser.add_argument("--max-prefix-gib", type=int, default=4)
    args = parser.parse_args()
    if not args.scene.replace("_", "").isalnum():
        raise ValueError("Invalid scene name")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    limit = args.max_prefix_gib * 1024**3
    if not 1 <= args.max_prefix_gib <= 6:
        raise ValueError("Bounded subset budget must be 1..6 GiB")
    cache = root / "cache/official_release_prefix.gz.partial"
    saved = cache.stat().st_size if cache.exists() else 0
    if shutil.disk_usage(root).free < max(0, limit - saved) + 2 * 1024**3:
        raise ValueError("Insufficient extraction/disk reserve")
    stream = ReleaseStream(cache, limit)
    records = []
    started = time.time()
    required = {
        "mesh.ply",
        "semantic.json",
        "semantic.bin",
        "preseg.json",
        "preseg.bin",
    }
    optional = {
        "habitat/mesh_semantic.ply",
        "habitat/info_semantic.json",
        "habitat/mesh_semantic.navmesh",
        "habitat/replica_stage.stage_config.json",
    }
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                parts = Path(member.name).parts
                if not parts or parts[0] != args.scene:
                    if required.issubset({r["relative_path"] for r in records}):
                        break
                    continue
                relative = "/".join(parts[1:])
                if relative not in required | optional or not member.isfile():
                    continue
                if member.size > 600 * 1024**2:
                    raise ValueError("Selected geometry member exceeds per-file budget")
                target = root / "raw" / member.name
                if not target.resolve().is_relative_to(root / "raw"):
                    raise ValueError("Unsafe native archive path")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size != member.size:
                        raise ValueError(
                            "Existing native asset size mismatch; preserved"
                        )
                    records.append(
                        {
                            "relative_path": relative,
                            "file": str(target),
                            "bytes": member.size,
                            "sha256": digest(target),
                            "reused": True,
                        }
                    )
                    continue
                temporary = target.with_suffix(target.suffix + ".partial")
                h = hashlib.sha256()
                count = 0
                with archive.extractfile(member) as source, temporary.open("xb") as out:
                    for data in iter(lambda: source.read(1024 * 1024), b""):
                        out.write(data)
                        h.update(data)
                        count += len(data)
                if count != member.size:
                    raise ValueError("Truncated selected member")
                temporary.replace(target)
                records.append(
                    {
                        "relative_path": relative,
                        "file": str(target),
                        "bytes": count,
                        "sha256": h.hexdigest(),
                        "reused": False,
                    }
                )
                print("Verified complete native member", relative, count, flush=True)
                if required.issubset({r["relative_path"] for r in records}) and any(
                    r["relative_path"] == "habitat/mesh_semantic.ply" for r in records
                ):
                    break
        complete = required.issubset({r["relative_path"] for r in records})
        result = {
            "dataset": "Replica",
            "scene": args.scene,
            "source": "https://github.com/facebookresearch/Replica-Dataset/releases/tag/v1.0",
            "status": "geometry_subset_downloaded" if complete else "incomplete",
            "files": records,
            "compressed_prefix_bytes": stream.position,
            "new_network_bytes": stream.network_bytes,
            "prefix_cache": str(cache),
            "full_archive_downloaded": False,
            "archive_crc": "Full gzip footer unavailable in bounded prefix; complete member sizes and local SHA256 recorded; verify mesh structure separately",
            "official_member_checksums_available": False,
            "license": "Replica Dataset Research Terms, noncommercial research/education",
            "missing": [
                "complete Ptex texture set",
                "source camera recordings and poses",
            ],
            "runtime_seconds": time.time() - started,
        }
        (root / "geometry_download.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if not complete:
            raise ValueError("Required native geometry files not reached within budget")
    except Exception as error:
        (root / "geometry_download.json").write_text(
            json.dumps(
                {
                    "dataset": "Replica",
                    "scene": args.scene,
                    "source": "https://github.com/facebookresearch/Replica-Dataset/releases/tag/v1.0",
                    "status": "incomplete",
                    "error": str(error),
                    "files": records,
                    "compressed_prefix_bytes": stream.position,
                    "new_network_bytes": stream.network_bytes,
                    "prefix_cache": str(cache),
                    "full_archive_downloaded": False,
                    "archive_crc_verified": False,
                    "missing_geometry": sorted(
                        required - {r["relative_path"] for r in records}
                    ),
                    "runtime_seconds": time.time() - started,
                },
                indent=2,
            )
        )
        raise
    finally:
        stream.close()


if __name__ == "__main__":
    main()
