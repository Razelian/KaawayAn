"""Create a self-contained KaawayAn source release zip."""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

EXCLUDED_DIR_OR_FILE_PATTERNS = [
    ".venv*",
    ".conda*",
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    ".git*",
    ".pytest_cache",
    "dist",
    "trials",
    "reports",
    "logs",
    "*.db",
    "*.db-wal",
    "*.db-shm",
    "*.wav",
    "dumps",
]

MODEL_WEIGHT_PATTERNS = [
    "*.pt",
    "*.pth",
    "*.bin",
    "*.onnx",
    "*.ckpt",
    "*.h5",
]


def should_exclude(rel_path: str, with_model: bool = False) -> bool:
    """Return True if rel_path matches any exclusion pattern."""
    parts = Path(rel_path).parts
    normalized = rel_path.replace("\\", "/").lower()
    if normalized.startswith("dashboard/ui_assets/") and filename_is_image(parts[-1] if parts else ""):
        return True
    for part in parts:
        for pat in EXCLUDED_DIR_OR_FILE_PATTERNS:
            if fnmatch.fnmatch(part.lower(), pat.lower()):
                return True
        if not with_model:
            for pat in MODEL_WEIGHT_PATTERNS:
                if fnmatch.fnmatch(part.lower(), pat.lower()):
                    return True
    filename = parts[-1] if parts else ""
    for pat in EXCLUDED_DIR_OR_FILE_PATTERNS:
        if fnmatch.fnmatch(filename.lower(), pat.lower()):
            return True
    if not with_model:
        for pat in MODEL_WEIGHT_PATTERNS:
            if fnmatch.fnmatch(filename.lower(), pat.lower()):
                return True
    return False


def filename_is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}


def collect_dashboard_files(dash_root: Path, with_model: bool) -> dict[str, Path]:
    """Collect source, runtime assets, and docs while excluding local artifacts."""
    files: dict[str, Path] = {}
    allowed_suffixes = {
        ".css", ".hex", ".html", ".ino", ".js", ".json", ".md", ".ps1", ".py",
        ".service", ".sh", ".svg", ".toml", ".txt", ".yaml", ".yml",
    }
    for current, dirs, names in os.walk(dash_root):
        current_path = Path(current)
        rel_dir = current_path.relative_to(dash_root).as_posix()
        dirs[:] = [
            name
            for name in dirs
            if not should_exclude(
                f"{name}" if rel_dir == "." else f"{rel_dir}/{name}",
                with_model,
            )
        ]
        for name in names:
            path = current_path / name
            if path.suffix.lower() not in allowed_suffixes and name not in {".gitignore", ".coveragerc"}:
                continue
            rel = path.relative_to(dash_root).as_posix()
            if name != "MANIFEST.txt" and not should_exclude(rel, with_model):
                files[rel] = path
    if with_model:
        for p in (dash_root / "model").glob("*"):
            if p.is_file() and any(fnmatch.fnmatch(p.name.lower(), pat) for pat in MODEL_WEIGHT_PATTERNS):
                files[p.relative_to(dash_root).as_posix()] = p
    return files


def collect_localization_files(loc_root: Path) -> dict[str, Path]:
    """Collect candidate files from LOCALIZATION root."""
    files: dict[str, Path] = {}

    # Backward-compatible helper for callers packaging an older sibling checkout.
    kw_dir = loc_root / "laptop" / "kaawayan"
    if kw_dir.is_dir():
        for p in kw_dir.rglob("*.py"):
            rel = p.relative_to(loc_root).as_posix()
            if not should_exclude(rel, with_model=False):
                files[rel] = p

    # Legacy sibling repository scripts.
    sc_dir = loc_root / "laptop" / "scripts"
    if sc_dir.is_dir():
        for p in sc_dir.glob("*.py"):
            rel = p.relative_to(loc_root).as_posix()
            if not should_exclude(rel, with_model=False):
                files[rel] = p

    # Legacy sibling repository packaging metadata.
    pyproj = loc_root / "laptop" / "pyproject.toml"
    if pyproj.is_file():
        files[pyproj.relative_to(loc_root).as_posix()] = pyproj

    # Root instruction/config files: CONTEXT.md, AGENTS.md, PROGRESS.md, requirements.txt
    for name in ["CONTEXT.md", "AGENTS.md", "PROGRESS.md", "requirements.txt"]:
        p = loc_root / name
        if p.is_file():
            files[name] = p

    # config/*.toml
    cfg_dir = loc_root / "config"
    if cfg_dir.is_dir():
        for p in cfg_dir.glob("*.toml"):
            if p.is_file():
                files[p.relative_to(loc_root).as_posix()] = p

    # Any firmware README
    fw_dir = loc_root / "firmware"
    if fw_dir.is_dir():
        for p in fw_dir.glob("README*"):
            if p.is_file():
                files[p.relative_to(loc_root).as_posix()] = p

    return files


def create_bundle(
    dash_root: Path,
    out_dir: Path,
    with_model: bool = False,
    loc_root: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_path = out_dir / f"KaawayAn_bundle_{stamp}.zip"

    all_files = collect_dashboard_files(dash_root, with_model)
    if loc_root is not None and loc_root.is_dir():
        loc_files = collect_localization_files(loc_root)
        # Prefix localization files under localization/ to avoid collision
        for rel, p in loc_files.items():
            all_files[f"localization/{rel}"] = p

    sorted_items = sorted(all_files.items(), key=lambda x: x[0])

    manifest_lines = []
    for rel_path, abs_path in sorted_items:
        size = abs_path.stat().st_size
        manifest_lines.append(f"{size:>10}  {rel_path}")
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("MANIFEST.txt", manifest_bytes)
        for rel_path, abs_path in sorted_items:
            zf.write(abs_path, arcname=rel_path)

    total_size = zip_path.stat().st_size
    print(f"Created bundle: {zip_path}")
    print(f"Total files included: {len(sorted_items) + 1} (including MANIFEST.txt)")
    print(f"Bundle size: {total_size} bytes ({total_size / (1024*1024):.2f} MB)")
    print("Files in bundle:")
    for rel, abs_p in sorted_items:
        print(f"  {rel} ({abs_p.stat().st_size} B)")

    if not with_model and total_size > 2 * 1024 * 1024:
        print(f"WARNING: Bundle size {total_size} bytes exceeds 2 MB limit!")

    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create KaawayAn distribution zip bundle")
    parser.add_argument("--with-model", action="store_true", help="Include model weights")
    parser.add_argument("--with-localization", default=None, help="Path to LOCALIZATION repo")
    parser.add_argument("--out", default="dist", help="Output directory (default: dist)")
    args = parser.parse_args(argv)

    dash_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = dash_root / out_dir

    loc_root = Path(args.with_localization) if args.with_localization else None
    if loc_root and not loc_root.is_absolute():
        loc_root = (dash_root / loc_root).resolve()

    create_bundle(
        dash_root=dash_root,
        out_dir=out_dir,
        with_model=args.with_model,
        loc_root=loc_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
