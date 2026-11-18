#!/usr/bin/env python3
"""
lmstudio_flags.py - add (or remove) the llama-server flags llama-gauges needs
to every LLM installed in LM Studio:

    --metrics --api-key-file <key file>

LM Studio (llama.cpp engine 2.x) starts one llama-server process per loaded
model on a random localhost port with a per-load API key. These flags enable
the Prometheus /metrics endpoint and add a second, stable API key that you own,
so llama-gauges can read /metrics and /slots. LM Studio's own key keeps working.

Where the flags are written:
  * regular models -> <lmstudio>/.internal/user-concrete-model-default-config/<model>.gguf.json
  * hub models     -> <lmstudio>/hub/models/<publisher>/<model>/model.yaml  (config.load)
    (hub / "virtual" models ignore the concrete config for load settings)

Every changed file is backed up once as <file>.bak-llama-gauges.
Reload a model in LM Studio for the change to take effect.

usage:
  python lmstudio_flags.py --create-key          # make the key file, add flags everywhere
  python lmstudio_flags.py --dry-run             # show what would change
  python lmstudio_flags.py --only gemma --only qwen
  python lmstudio_flags.py --remove              # take the flags out again
  python lmstudio_flags.py --list                # show models and their current overrides
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import sys
from pathlib import Path

FIELD = "llm.load.llama.argumentsOverride"
BACKUP_SUFFIX = ".bak-llama-gauges"
EMBED_RE = re.compile(r"embed|bge|rerank|nomic|e5-|gte-|minilm", re.I)


def default_lmstudio_dir() -> Path:
    env = os.environ.get("LMSTUDIO_HOME")
    return Path(env).expanduser() if env else Path.home() / ".lmstudio"


# --------------------------------------------------------------------------- #
# merge / remove logic on a load.fields list
# --------------------------------------------------------------------------- #
def wanted_flags(key_file: Path) -> list[dict]:
    return [{"key": "--metrics", "value": ""},
            {"key": "--api-key-file", "value": key_file.as_posix()}]


def add_flags(fields: list, flags: list[dict]) -> str:
    for f in fields:
        if f.get("key") == FIELD:
            v = f.setdefault("value", {})
            params = v.setdefault("overrideParameters", [])
            fixed = 0
            for p in params:  # a leading/trailing space in a key makes llama-server refuse to start
                k = p.get("key", "")
                if k != k.strip():
                    p["key"] = k.strip()
                    fixed += 1
            changed = []
            for fl in flags:
                cur = next((p for p in params if p.get("key") == fl["key"]), None)
                if cur is None:
                    params.append(dict(fl))
                    changed.append(f"+{fl['key']}")
                elif cur.get("value", "") != fl["value"]:
                    cur["value"] = fl["value"]
                    changed.append(f"~{fl['key']}")
            was_enabled = v.get("enabled") is True
            v["enabled"] = True
            v.setdefault("disabledParameters", [])
            v.setdefault("excludeAllConfig", False)
            if fixed:
                changed.append(f"fixed {fixed} key(s) with stray spaces")
            if not changed and was_enabled:
                return ""
            return ", ".join(changed) or "enabled overrides"
    fields.append({"key": FIELD, "value": {"enabled": True, "disabledParameters": [],
                                           "overrideParameters": [dict(fl) for fl in flags],
                                           "excludeAllConfig": False}})
    return "added"


def remove_flags(fields: list, flags: list[dict]) -> str:
    names = {fl["key"] for fl in flags}
    for f in fields:
        if f.get("key") == FIELD:
            params = f.get("value", {}).get("overrideParameters", [])
            keep = [p for p in params if p.get("key", "").strip() not in names]
            if len(keep) == len(params):
                return ""
            f["value"]["overrideParameters"] = keep
            if not keep:
                fields.remove(f)
            return f"removed {len(params) - len(keep)} flag(s)"
    return ""


def describe(fields: list) -> str:
    for f in fields:
        if f.get("key") == FIELD:
            v = f.get("value", {})
            ps = " ".join(f"{p.get('key')}{(' ' + p['value']) if p.get('value') else ''}"
                          for p in v.get("overrideParameters", []))
            return ("" if v.get("enabled") else "(disabled) ") + (ps or "(empty)")
    return "-"


# --------------------------------------------------------------------------- #
# model discovery
# --------------------------------------------------------------------------- #
def concrete_targets(root: Path):
    base = root / ".internal" / "user-concrete-model-default-config"
    if not base.is_dir():
        return
    for cfg in sorted(base.rglob("*.gguf.json")):
        rel = cfg.relative_to(base).as_posix()[: -len(".json")]
        yield rel, cfg, (root / "models" / rel).exists()


def hub_targets(root: Path):
    base = root / "hub" / "models"
    if not base.is_dir():
        return
    for y in sorted(base.glob("*/*/model.yaml")):
        text = y.read_text(encoding="utf-8")
        m = re.search(r"^\s*domain:\s*(\S+)", text, re.M)
        domain = m.group(1) if m else "llm"
        yield f"{y.parent.parent.name}/{y.parent.name}", y, domain


def read_hub_load(lines: list[str]):
    """Return (index of 'config:' or None, index of '  load:' line or None)."""
    try:
        ci = lines.index("config:")
    except ValueError:
        return None, None
    for j in range(ci + 1, len(lines)):
        ln = lines[j]
        if ln and not ln.startswith(" "):  # next top-level key
            break
        if ln.startswith("  load:"):
            return ci, j
    return ci, None


# --------------------------------------------------------------------------- #
def backup(path: Path, dry: bool) -> None:
    b = path.with_name(path.name + BACKUP_SUFFIX)
    if not dry and not b.exists():
        shutil.copy2(path, b)


def main() -> int:
    ap = argparse.ArgumentParser(description="Add/remove llama-gauges flags in LM Studio model configs.")
    ap.add_argument("--lmstudio-dir", type=Path, default=default_lmstudio_dir(),
                    help="LM Studio data dir (default: $LMSTUDIO_HOME or ~/.lmstudio)")
    ap.add_argument("--key-file", type=Path, help="API key file (default: <lmstudio-dir>/metrics-key.txt)")
    ap.add_argument("--create-key", action="store_true", help="create the key file if it does not exist")
    ap.add_argument("--remove", action="store_true", help="remove the flags instead of adding them")
    ap.add_argument("--only", action="append", default=[], help="only models whose path contains this (repeatable)")
    ap.add_argument("--include-embeddings", action="store_true", help="also touch embedding / reranker models")
    ap.add_argument("--include-missing", action="store_true",
                    help="also touch configs whose .gguf is no longer on disk")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    ap.add_argument("--list", action="store_true", help="list models and their current argument overrides")
    a = ap.parse_args()

    root = a.lmstudio_dir.expanduser()
    if not root.is_dir():
        print(f"LM Studio dir not found: {root}  (use --lmstudio-dir)", file=sys.stderr)
        return 2
    key_file = (a.key_file or root / "metrics-key.txt").expanduser().resolve()
    flags = wanted_flags(key_file)

    if not a.list and not a.remove:
        if not key_file.exists():
            if not a.create_key:
                print(f"key file missing: {key_file}\n  run again with --create-key, or pass --key-file",
                      file=sys.stderr)
                return 2
            if not a.dry_run:
                key_file.parent.mkdir(parents=True, exist_ok=True)
                key_file.write_text(secrets.token_urlsafe(24), encoding="utf-8")
                print(f"created key file {key_file}")

    def selected(name: str) -> bool:
        if a.only and not any(o.lower() in name.lower() for o in a.only):
            return False
        return a.include_embeddings or not EMBED_RE.search(name)

    op = remove_flags if a.remove else add_flags
    tag = "dry " if a.dry_run else ""
    touched = 0

    for rel, cfg, on_disk in concrete_targets(root):
        if not selected(rel):
            continue
        data = json.loads(cfg.read_text(encoding="utf-8"))
        fields = data.setdefault("load", {}).setdefault("fields", [])
        if a.list:
            print(f"model  {rel}{'' if on_disk else '  [not on disk]'}\n       {describe(fields)}")
            continue
        if not on_disk and not a.include_missing:
            continue
        status = op(fields, flags)
        if status:
            touched += 1
            if not a.dry_run:
                backup(cfg, False)
                cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"{tag}model  {rel}: {status}")

    for name, y, domain in hub_targets(root):
        if not selected(name) or (domain != "llm" and not a.include_embeddings):
            continue
        lines = y.read_text(encoding="utf-8").split("\n")
        ci, li = read_hub_load(lines)
        load = {"fields": []}
        if li is not None:
            try:
                load = json.loads(lines[li][len("  load:"):].strip())
            except json.JSONDecodeError:
                print(f"skip   {name}: config.load in model.yaml is not single-line JSON - edit it by hand")
                continue
        fields = load.setdefault("fields", [])
        if a.list:
            print(f"hub    {name}\n       {describe(fields)}")
            continue
        status = op(fields, flags)
        if not status:
            continue
        new_line = "  load: " + json.dumps(load)
        if li is not None:
            lines[li] = new_line
        elif ci is not None:
            lines.insert(ci + 1, new_line)
        else:
            if lines and lines[-1] == "":
                lines.pop()
            lines += ["config:", new_line, ""]
        touched += 1
        if not a.dry_run:
            backup(y, False)
            y.write_text("\n".join(lines), encoding="utf-8")
        print(f"{tag}hub    {name}: {status}")

    if not a.list:
        print(f"\n{touched} file(s) {'would change' if a.dry_run else 'changed'}."
              + ("" if a.dry_run or not touched else "  Reload the model(s) in LM Studio."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
