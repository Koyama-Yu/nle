#!/usr/bin/env python
import argparse
import json
from pathlib import Path


INVENTORY_KEYS = (
    "inv_pickups_by_name",
    "inv_pickups_by_class",
    "inv_uses_by_action",
    "inv_uses_by_name",
    "inv_uses_by_class",
)


def parse_xlog_line(xlog_path: Path, ttyrec_name: str):
    with open(xlog_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = {}
            for part in line.split("\t"):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                fields[key] = value
            if fields.get("ttyrecname") == ttyrec_name:
                return fields
    return {}


def extract_inventory_metadata(fields):
    metadata = {}
    for key in INVENTORY_KEYS:
        raw = fields.get(key)
        if not raw:
            continue
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = {"_raw": raw, "_error": "invalid json"}
    return metadata


def derive_xlog_path(ttyrec_path: Path) -> Path:
    # ttyrec path looks like .../nle.<pid>.0.ttyrec3.bz2
    base = ".".join(ttyrec_path.name.split(".")[:2])
    return ttyrec_path.parent / f"{base}.xlogfile"


def main():
    parser = argparse.ArgumentParser(
        description="指定した ttyrec/xlogfile から inventory metadata を確認します。"
    )
    parser.add_argument(
        "--ttyrec",
        required=True,
        help="対象の ttyrec ファイルへのパス",
    )
    parser.add_argument(
        "--xlog",
        default=None,
        help="対応する xlogfile のパス（省略時は ttyrec から推測）",
    )
    args = parser.parse_args()

    ttyrec_path = Path(args.ttyrec).expanduser()
    if not ttyrec_path.exists():
        raise FileNotFoundError(f"ttyrec が見つかりません: {ttyrec_path}")

    xlog_path = (
        Path(args.xlog).expanduser()
        if args.xlog
        else derive_xlog_path(ttyrec_path)
    )
    if not xlog_path.exists():
        raise FileNotFoundError(f"xlogfile が見つかりません: {xlog_path}")

    fields = parse_xlog_line(xlog_path, ttyrec_path.name)
    if not fields:
        raise RuntimeError(
            f"{xlog_path} 内に ttyrecname={ttyrec_path.name} の行が見つかりません。"
        )

    metadata = extract_inventory_metadata(fields)
    print(f"ttyrec: {ttyrec_path}")
    print(f"xlog : {xlog_path}")
    if metadata:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
    else:
        print("inventory metadata のフィールドが見つかりませんでした。")


if __name__ == "__main__":
    main()
