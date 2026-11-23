"""Extract official archives once, into a fresh ignored directory; never overwrite."""
import argparse
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def checked_members(archive):
    seen = set()
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or ":" in name
                or not path.parts or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError(f"Güvensiz ZIP yolu: {name}")
        key = name.rstrip("/").casefold()
        if key in seen:
            raise ValueError(f"Çakışan ZIP yolu: {name}")
        seen.add(key)
        yield info, path


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Hedef zaten var; hiçbir şey değişmedi: {destination}")
    archives = [source / f"uavid_{split}.zip" for split in ("train", "val")]
    # Preflight both archives before any extraction.
    plans = []
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            members = list(checked_members(archive))
            expected = path.stem
            if any(member.parts[0] != expected for _, member in members):
                raise ValueError(f"Beklenmeyen üst klasör: {path.name}")
            plans.append((path, members))
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = sum(info.file_size for _, members in plans for info, _ in members)
    if shutil.disk_usage(destination.parent).free < required + 512 * 1024**2:
        raise OSError("Arşivleri açmak için yeterli boş alan yok.")
    # A unique staging folder prevents partial data from being used by notebooks.
    staging = Path(tempfile.mkdtemp(prefix=".uavid-partial-", dir=destination.parent))
    try:
        for path, members in plans:
            with zipfile.ZipFile(path) as archive:
                for info, member in members:
                    target = staging.joinpath(*member.parts).resolve()
                    if not target.is_relative_to(staging):
                        raise ValueError("ZIP hedefi sınır dışına çıktı.")
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as reader, target.open("xb") as writer:
                            shutil.copyfileobj(reader, writer)
        readme = source / "readme.txt"
        if readme.is_file():
            shutil.copyfile(readme, staging / "readme.txt")
        # Windows rename fails if destination exists; do not merge directories.
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    except Exception:
        # Keep staged files for inspection; no recursive deletion or overwrite.
        print(f"Tamamlanmamış geçici klasör: {staging}")
        raise
    print(f"Veri hazır: {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
