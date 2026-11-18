#!/usr/bin/env python3
"""
llama_gauges.py - interactive terminal dashboard for llama-server / LM Studio.

Finds every running llama-server process (including the ones LM Studio spawns
per loaded model), reads its port and --api-key-file from the process command
line, and polls /metrics and /slots. Hardware comes from NVML (nvidia-ml-py,
falls back to nvidia-smi) and psutil.

Views   1 Overview   2 Slots   3 Requests   4 Hardware
Keys    1-4 view · Tab / ←→ switch model · Space freeze · r refresh now
        t theme · c clear history · ? help · q quit

Requirements : pip install rich psutil nvidia-ml-py   (nvidia-ml-py is optional)
Server flags : --metrics, plus --api-key-file <file> when the server uses an
               API key (LM Studio's always do). The server's own --api-key is
               never touched.

Usage:
  python llama_gauges.py                       # full-screen, refresh every 1 s
  python llama_gauges.py -i 0.5                # faster polling
  python llama_gauges.py --no-screen           # draw inline instead of full-screen
  python llama_gauges.py --port 8080 --key-file key.txt   # just this server
  python llama_gauges.py --once --view 1       # print one snapshot and exit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

__version__ = "0.2.0"

import psutil
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import pynvml  # nvidia-ml-py
except ImportError:  # pragma: no cover
    pynvml = None

SPARK = "▁▂▃▄▅▆▇█"
PARTIAL = " ▏▎▍▌▋▊▉"
METRIC_RE = re.compile(r"^llamacpp:(\w+)(?:\{([^}]*)\})?\s+(\S+)$")
GB = 1024 ** 3
PCIE_LANE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}
VIEWS = ["Overview", "Slots", "Requests", "Hardware"]

THEMES = [
    ("ocean", dict(accent="bright_cyan", gen="cyan", pp="magenta", gpu="green3", vram="gold1",
                   temp="orange_red1", power="dark_orange", pcie="deep_sky_blue1", cpu="dodger_blue1",
                   disk="medium_purple1", ram="steel_blue1", draft="spring_green2", slots="yellow",
                   ok="green3", warn="gold1", bad="red1", text="grey93", dim="grey50", empty="grey23",
                   border="grey35", badge="black on bright_cyan")),
    ("ember", dict(accent="orange1", gen="orange1", pp="hot_pink", gpu="yellow2", vram="light_goldenrod1",
                   temp="red1", power="dark_orange3", pcie="sandy_brown", cpu="gold3", disk="light_salmon1",
                   ram="khaki1", draft="chartreuse2", slots="orange3", ok="yellow2", warn="orange1",
                   bad="red1", text="grey93", dim="grey50", empty="grey23", border="grey30",
                   badge="black on orange1")),
    ("forest", dict(accent="spring_green1", gen="spring_green1", pp="orchid", gpu="green_yellow",
                    vram="dark_sea_green2", temp="salmon1", power="tan", pcie="aquamarine1",
                    cpu="pale_green1", disk="plum2", ram="dark_sea_green", draft="cyan2", slots="khaki1",
                    ok="green1", warn="khaki1", bad="red1", text="grey93", dim="grey50", empty="grey23",
                    border="grey30", badge="black on spring_green1")),
    ("mono", dict(accent="white", gen="white", pp="grey85", gpu="grey85", vram="grey85", temp="grey85",
                  power="grey85", pcie="grey85", cpu="grey85", disk="grey85", ram="grey85", draft="grey85",
                  slots="grey85", ok="white", warn="white", bad="bold white", text="grey93", dim="grey50",
                  empty="grey23", border="grey35", badge="black on white")),
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def bar(frac: float, width: int, style: str, empty: str = "grey23", char: str = "█") -> Text:
    """Horizontal gauge with 1/8-cell resolution."""
    width = max(1, width)
    if frac != frac:  # NaN
        frac = 0.0
    frac = max(0.0, min(1.0, frac))
    cells = frac * width
    full = int(cells)
    t = Text()
    t.append(char * full, style=style)
    if full < width:
        part = int((cells - full) * 8)
        if part:
            t.append(PARTIAL[part], style=style)
        t.append("━" * (width - full - (1 if part else 0)), style=empty)
    return t


def spark(values, top: float, width: int, style: str, empty: str = "grey23") -> Text:
    width = max(1, width)
    vals = list(values)[-width:]
    top = top if top and top > 0 else 1.0
    t = Text()
    pad = width - len(vals)
    if pad > 0:
        t.append("▁" * pad, style=empty)
    for v in vals:
        if v is None or v <= 0:
            t.append("▁", style=empty)
        else:
            t.append(SPARK[min(7, int(v / top * 7.999))], style=style)
    return t


def fint(v) -> str:
    return f"{int(v):,}" if v is not None else "-"


def fk(v) -> str:
    v = float(v or 0)
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1000:
        if v % 1024 == 0:
            return f"{v / 1024:.0f}K"
        return f"{v / 1000:.0f}k" if v % 1000 == 0 else f"{v / 1000:.1f}k"
    return f"{v:.0f}"


def ago(ts: float) -> str:
    d = time.time() - ts
    if d < 60:
        return f"{d:.0f}s ago"
    if d < 3600:
        return f"{d / 60:.0f}m ago"
    return f"{d / 3600:.1f}h ago"


def safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def level(frac: float, base: str, th: dict, warn: float = 0.85, bad: float = 0.95) -> str:
    return th["bad"] if frac >= bad else th["warn"] if frac >= warn else base


def read_key(path) -> str | None:
    try:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                return line.strip()
    except OSError:
        pass
    return None


def http_get(url: str, key: str | None, timeout: float = 1.5) -> str:
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_metrics(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = METRIC_RE.match(line.strip())
        if not m:
            continue
        name, labels, val = m.groups()
        v = safe_float(val)
        if v is None:
            continue
        if labels:
            pm = re.search(r'position="(\d+)"', labels)
            if pm:
                out.setdefault(name, {})[int(pm.group(1))] = v
                continue
            name = f"{name}{{{labels}}}"
        out[name] = v
    return out


def opts_from_cmdline(cmd: list[str]) -> dict:
    opts, i = {}, 1
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("-"):
            val = None
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                val = cmd[i + 1]
                i += 1
            opts[tok] = val
        i += 1
    return opts


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def discover(only_ports: set[int]) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if not (p.info["name"] or "").lower().startswith("llama-server"):
                continue
            cmd = p.info["cmdline"] or []
        except (psutil.Error, OSError):
            continue
        o = opts_from_cmdline(cmd)
        port_s = o.get("--port")
        if not port_s or not port_s.isdigit():
            continue
        port = int(port_s)
        if only_ports and port not in only_ports:
            continue
        host = o.get("--host") or "127.0.0.1"
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        found[port] = dict(
            pid=p.info["pid"], host=host, port=port,
            model=Path(o.get("--model") or o.get("-m") or "?").stem,
            ctx=o.get("--ctx-size") or o.get("-c"),
            parallel=o.get("--parallel") or o.get("-np"),
            key_file=o.get("--api-key-file"),
            metrics_flag="--metrics" in o,
            spec=o.get("--spec-type"),
            ncpumoe=o.get("--n-cpu-moe"),
            runtime=Path(cmd[0]).parent.name if cmd else "",
        )
    for port in only_ports - found.keys():  # manual servers not seen as processes
        found[port] = dict(pid=None, host="127.0.0.1", port=port, model=f"server :{port}", ctx=None,
                           parallel=None, key_file=None, metrics_flag=True, spec=None, ncpumoe=None,
                           runtime="manual")
    return found


# --------------------------------------------------------------------------- #
# per-server state
# --------------------------------------------------------------------------- #
class Server:
    def __init__(self, info: dict, key_override: str | None):
        self.info = info
        self.key_override = key_override
        self.gen_hist: deque[float] = deque(maxlen=600)
        self.pp_hist: deque[float] = deque(maxlen=600)
        self.gen_rate = 0.0
        self.pp_rate = 0.0
        self.peak_gen = 0.0
        self.peak_pp = 0.0
        self.slot_rate: dict = {}
        self.prev = None            # (monotonic t, {slot: (task, n_decoded, n_pp)})
        self.prev_m: dict | None = None
        self.metrics: dict = {}
        self.slots: list = []
        self.active: dict = {}      # (slot, task) -> in-flight request
        self.requests: deque = deque(maxlen=500)
        self.pending: list = []     # finished requests waiting for exact /metrics numbers
        self.last_gen: dict | None = None
        self.error: str | None = None
        self.slots_error: str | None = None

    @property
    def base(self) -> str:
        return f"http://{self.info['host']}:{self.info['port']}"

    def key(self) -> str | None:
        path = self.key_override or self.info.get("key_file")
        return read_key(path) if path else None

    def rss(self) -> float | None:
        try:
            return psutil.Process(self.info["pid"]).memory_info().rss if self.info["pid"] else None
        except (psutil.Error, OSError):
            return None

    def clear(self) -> None:
        self.gen_hist.clear()
        self.pp_hist.clear()
        self.peak_gen = self.peak_pp = 0.0
        self.requests.clear()

    # ---- derived ---------------------------------------------------------
    @staticmethod
    def next_token(slot: dict) -> dict:
        nt = slot.get("next_token")
        if isinstance(nt, list):
            return nt[0] if nt else {}
        return nt if isinstance(nt, dict) else {}

    def busy_slots(self) -> list:
        return [s for s in self.slots if s.get("is_processing")]

    def n_slots(self) -> int:
        return len(self.slots) or int(self.info.get("parallel") or 0) or 1

    def state(self) -> str:
        if self.error:
            return "error"
        if self.metrics.get("requests_deferred", 0) > 0:
            return "queued"
        busy = self.busy_slots()
        if any(self.next_token(s).get("n_decoded") for s in busy):
            return "generating"
        if busy:
            return "reading"
        return "idle"

    def primary(self) -> dict | None:
        """The active slot shown in the progress row: the one furthest along."""
        busy = self.busy_slots()
        if not busy:
            return None
        return max(busy, key=lambda s: (s.get("n_prompt_tokens", 0) or 0)
                   + (self.next_token(s).get("n_decoded", 0) or 0))

    def progress(self) -> tuple[float, str] | None:
        s = self.primary()
        if not s:
            return None
        nt = self.next_token(s)
        dec = nt.get("n_decoded", 0) or 0
        n_prompt = s.get("n_prompt_tokens", 0) or 0
        cache = s.get("n_prompt_tokens_cache", 0) or 0
        if not dec:
            done = (s.get("n_prompt_tokens_processed", 0) or 0) + cache
            frac = done / n_prompt if n_prompt else 0
            return frac, f"{fint(done)} / {fint(n_prompt)} prompt tokens · {frac * 100:.0f}%"
        remain = nt.get("n_remain")
        if remain is not None and remain >= 0:
            frac = dec / (dec + remain) if dec + remain else 0
            return frac, f"{fint(dec)} / {fint(dec + remain)} output tokens · {frac * 100:.0f}%"
        n_ctx = s.get("n_ctx") or 0
        used = n_prompt + dec
        return (used / n_ctx if n_ctx else 0), f"{fint(dec)} tokens out · ctx {fint(used)} / {fk(n_ctx)}"

    # ---- polling ---------------------------------------------------------
    def poll(self) -> None:
        key = self.key()
        now = time.monotonic()
        self.error = None
        try:
            self.metrics = parse_metrics(http_get(self.base + "/metrics", key))
        except urllib.error.HTTPError as e:
            self.metrics = {}
            self.error = {
                401: "401 unauthorized - add  --api-key-file <file>  to the model's custom flags (or pass --key-file)",
                501: "501 metrics disabled - add  --metrics  to the model's custom flags",
                503: "503 model is loading...",
            }.get(e.code, f"HTTP {e.code} on /metrics")
        except (urllib.error.URLError, OSError) as e:
            self.metrics = {}
            self.error = f"unreachable: {getattr(e, 'reason', e)}"
        try:
            data = json.loads(http_get(self.base + "/slots", key))
            self.slots = data if isinstance(data, list) else []
            self.slots_error = None
        except urllib.error.HTTPError as e:
            self.slots, self.slots_error = [], f"/slots HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError):
            self.slots, self.slots_error = [], "/slots unavailable"
        self._live_rates(now)
        self._track_requests(time.time())
        self._finished_from_metrics()

    def _live_rates(self, now: float) -> None:
        cur = {}
        for s in self.busy_slots():
            cur[s.get("id")] = (s.get("id_task"), self.next_token(s).get("n_decoded", 0) or 0,
                                s.get("n_prompt_tokens_processed", 0) or 0)
        gen = pp = 0.0
        rates = {}
        if self.prev:
            t0, before = self.prev
            dt = now - t0
            if dt > 0:
                for sid, (task, dec, ppn) in cur.items():
                    b = before.get(sid)
                    if b and b[0] == task:
                        g, p = max(0, dec - b[1]), max(0, ppn - b[2])
                    else:  # task started since the last poll
                        g, p = dec, ppn
                    rates[sid] = g / dt
                    gen += g / dt
                    pp += p / dt
        self.prev = (now, cur)
        a = 0.6
        for sid, r in rates.items():
            old = self.slot_rate.get(sid)
            self.slot_rate[sid] = a * r + (1 - a) * old if old else r
        for sid in list(self.slot_rate):
            if sid not in cur:
                del self.slot_rate[sid]
        if cur:
            self.gen_rate = a * gen + (1 - a) * self.gen_rate if self.gen_rate and gen else gen
            self.pp_rate = a * pp + (1 - a) * self.pp_rate if self.pp_rate and pp else pp
        else:
            self.gen_rate = self.pp_rate = 0.0
        self.peak_gen = max(self.peak_gen, self.gen_rate)
        self.peak_pp = max(self.peak_pp, self.pp_rate)
        self.gen_hist.append(self.gen_rate)
        self.pp_hist.append(self.pp_rate)

    def _track_requests(self, now: float) -> None:
        seen = set()
        for s in self.busy_slots():
            k = (s.get("id"), s.get("id_task"))
            seen.add(k)
            nt = self.next_token(s)
            dec = nt.get("n_decoded", 0) or 0
            r = self.active.get(k)
            if r is None:
                r = self.active[k] = dict(slot=s.get("id"), task=s.get("id_task"), start=now,
                                          gen_start=None, dec0=0, out=0)
            r.update(prompt=s.get("n_prompt_tokens", 0) or 0, reused=s.get("n_prompt_tokens_cache", 0) or 0,
                     pp=s.get("n_prompt_tokens_processed", 0) or 0, remain=nt.get("n_remain"),
                     n_ctx=s.get("n_ctx"), last=now, out=dec)
            if dec and r["gen_start"] is None:
                r["gen_start"], r["dec0"] = now, dec
        for k in [k for k in self.active if k not in seen]:
            r = self.active.pop(k)
            gen_t = (r["last"] - r["gen_start"]) if r["gen_start"] else 0
            tps = (r["out"] - r["dec0"]) / gen_t if gen_t > 0 and r["out"] > r["dec0"] else None
            rec = dict(time=r["start"], status="Done", slot=r["slot"], prompt=r["prompt"], reused=r["reused"],
                       out=r["out"], tps=tps, dur=max(0.0, r["last"] - r["start"]), exact=False, wait=0)
            self.requests.appendleft(rec)
            self.pending.append(rec)

    def _finished_from_metrics(self) -> None:
        """Counters in /metrics only move when a prefill / generation finishes."""
        m = self.metrics
        if not m:
            return
        p = self.prev_m
        if p:
            def d(k):
                a, b = m.get(k, 0), p.get(k, 0)
                return a - b if isinstance(a, float) and isinstance(b, float) else 0

            d_tok, d_sec = d("tokens_predicted_total"), d("tokens_predicted_seconds_total")
            if d_tok > 0:
                d_draft = d("spec_decode_num_draft_tokens_total")
                d_acc = d("spec_decode_num_accepted_tokens_total")
                self.last_gen = dict(when=time.time(), tokens=d_tok, tps=d_tok / d_sec if d_sec > 0 else 0.0,
                                     accept=d_acc / d_draft if d_draft > 0 else None)
                if len(self.pending) == 1:  # exactly one request finished: use the exact numbers
                    rec = self.pending[0]
                    rec.update(out=int(d_tok), exact=True, accept=self.last_gen["accept"])
                    if d_sec > 0:
                        rec["tps"] = d_tok / d_sec
                    self.pending.clear()
        for rec in self.pending:
            rec["wait"] += 1
        self.pending = [r for r in self.pending if r["wait"] < 3]
        self.prev_m = m


# --------------------------------------------------------------------------- #
# hardware sampler (background thread)
# --------------------------------------------------------------------------- #
class Hardware(threading.Thread):
    def __init__(self, period: float = 1.0, gpu: bool = True):
        super().__init__(daemon=True)
        self.period = period
        self.want_gpu = gpu
        self.lock = threading.Lock()
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=600))
        self.cores = psutil.cpu_count(logical=False) or 0
        self.threads = psutil.cpu_count(logical=True) or 0
        self.cpu = 0.0
        self.percpu: list[float] = []
        self.ram = (0.0, 1.0)
        self.disk_r = self.disk_w = 0.0
        self.gpus: list[dict] = []
        self.gpu_err: str | None = None
        self.gpu_src = None
        self.samples = 0

    def clear(self) -> None:
        with self.lock:
            for h in self.hist.values():
                h.clear()

    def _init_gpu(self) -> None:
        if not self.want_gpu:
            return
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.gpu_src = "nvml"
                return
            except Exception as e:  # noqa: BLE001
                self.gpu_err = f"NVML: {e}"
        if shutil.which("nvidia-smi"):
            self.gpu_src = "smi"
        elif not self.gpu_err:
            self.gpu_err = "no NVIDIA GPU tooling found"

    def _read_nvml(self) -> list[dict]:
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)

            def q(fn, *a):
                try:
                    return fn(h, *a)
                except Exception:  # noqa: BLE001
                    return None

            name = q(pynvml.nvmlDeviceGetName) or "GPU"
            if isinstance(name, bytes):
                name = name.decode()
            u = q(pynvml.nvmlDeviceGetUtilizationRates)
            mem = q(pynvml.nvmlDeviceGetMemoryInfo)
            pw, pl = q(pynvml.nvmlDeviceGetPowerUsage), q(pynvml.nvmlDeviceGetEnforcedPowerLimit)
            rx = q(pynvml.nvmlDeviceGetPcieThroughput, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            tx = q(pynvml.nvmlDeviceGetPcieThroughput, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            out.append(dict(
                name=name.replace("NVIDIA ", "").replace("GeForce ", ""),
                util=u.gpu if u else None,
                used=mem.used / GB if mem else None, total=mem.total / GB if mem else None,
                temp=q(pynvml.nvmlDeviceGetTemperature, pynvml.NVML_TEMPERATURE_GPU),
                power=pw / 1000 if pw is not None else None, limit=pl / 1000 if pl else None,
                gen=q(pynvml.nvmlDeviceGetCurrPcieLinkGeneration),
                max_gen=q(pynvml.nvmlDeviceGetMaxPcieLinkGeneration),
                width=q(pynvml.nvmlDeviceGetCurrPcieLinkWidth),
                rx=rx / 1024 if rx is not None else None, tx=tx / 1024 if tx is not None else None,  # MB/s
                sm=q(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_SM),
                memclk=q(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_MEM),
                fan=q(pynvml.nvmlDeviceGetFanSpeed),
            ))
        return out

    def _read_smi(self) -> list[dict]:
        fields = ("name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
                  "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,clocks.sm,fan.speed")
        res = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        out = []
        for line in res.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 12:
                continue
            sf = [safe_float(x) for x in f]
            out.append(dict(name=f[0].replace("NVIDIA ", "").replace("GeForce ", ""), util=sf[1],
                            used=sf[2] / 1024 if sf[2] else None, total=sf[3] / 1024 if sf[3] else None,
                            temp=sf[4], power=sf[5], limit=sf[6],
                            gen=int(sf[7]) if sf[7] else None, max_gen=int(sf[8]) if sf[8] else None,
                            width=int(sf[9]) if sf[9] else None, rx=None, tx=None, sm=sf[10], memclk=None,
                            fan=sf[11]))
        return out

    def run(self) -> None:
        self._init_gpu()
        psutil.cpu_percent(percpu=True)
        prev_disk, prev_t = psutil.disk_io_counters(), time.monotonic()
        while True:
            gpus = []
            try:
                if self.gpu_src == "nvml":
                    gpus = self._read_nvml()
                elif self.gpu_src == "smi":
                    gpus = self._read_smi()
            except Exception as e:  # noqa: BLE001
                self.gpu_err = str(e)
            with self.lock:
                self.gpus = gpus or self.gpus
            time.sleep(self.period)
            percpu = psutil.cpu_percent(percpu=True)
            vm = psutil.virtual_memory()
            disk, now = psutil.disk_io_counters(), time.monotonic()
            dt = max(1e-3, now - prev_t)
            dr = (disk.read_bytes - prev_disk.read_bytes) / dt / 1e6 if disk and prev_disk else 0.0
            dw = (disk.write_bytes - prev_disk.write_bytes) / dt / 1e6 if disk and prev_disk else 0.0
            prev_disk, prev_t = disk, now
            with self.lock:
                self.samples += 1
                self.percpu = percpu
                self.cpu = sum(percpu) / len(percpu) if percpu else 0.0
                self.ram = (vm.used / GB, vm.total / GB)
                self.disk_r, self.disk_w = max(0.0, dr), max(0.0, dw)
                h = self.hist
                h["cpu"].append(self.cpu)
                h["ram"].append(vm.percent)
                h["disk_r"].append(self.disk_r)
                h["disk_w"].append(self.disk_w)
                for i, g in enumerate(self.gpus):
                    for k in ("util", "temp", "power", "rx", "tx", "sm", "fan"):
                        h[f"g{i}.{k}"].append(g.get(k) or 0)
                    h[f"g{i}.vram"].append((g["used"] / g["total"] * 100) if g.get("used") and g.get("total") else 0)


# --------------------------------------------------------------------------- #
# keyboard (non-blocking, Windows + POSIX)
# --------------------------------------------------------------------------- #
class Keys:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._old = None
        if self.enabled and os.name != "nt":
            import termios
            import tty
            self._termios = termios
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def close(self) -> None:
        if self._old is not None:
            self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old)

    def read(self) -> list[str]:
        if not self.enabled:
            return []
        out = []
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    ch2 = msvcrt.getwch()
                    out.append({"K": "left", "M": "right", "H": "up", "P": "down"}.get(ch2, ""))
                else:
                    out.append(ch)
        else:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    seq = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.01)[0] else ""
                    out.append({"[D": "left", "[C": "right", "[A": "up", "[B": "down"}.get(seq, "esc"))
                else:
                    out.append(ch)
        return out


# --------------------------------------------------------------------------- #
# the app: state + rendering
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, args, console: Console):
        self.args = args
        self.console = console
        self.servers: dict[int, Server] = {}
        self.order: list[int] = []
        self.idx = 0
        self.view = max(0, min(len(VIEWS) - 1, (args.view or 1) - 1))
        self.theme = 0
        self.frozen = False
        self.help = False
        self.last_disc = 0.0
        self.only = set(args.port or [])
        self.hw = Hardware(period=max(0.5, args.interval), gpu=not args.no_gpu)
        self.hw.start()

    @property
    def th(self) -> dict:
        return THEMES[self.theme][1]

    @property
    def cur(self) -> Server | None:
        if not self.order:
            return None
        self.idx %= len(self.order)
        return self.servers.get(self.order[self.idx])

    # ---- state -----------------------------------------------------------
    def tick(self) -> None:
        if not self.servers or time.monotonic() - self.last_disc > 3:
            found = discover(self.only)
            for port, info in found.items():
                if port not in self.servers or self.servers[port].info["pid"] != info["pid"]:
                    self.servers[port] = Server(info, self.args.key_file)
                else:
                    self.servers[port].info = info
            for port in list(self.servers):
                if port not in found:
                    del self.servers[port]
            self.order = sorted(self.servers)
            self.last_disc = time.monotonic()
        for s in list(self.servers.values()):
            s.poll()

    def handle(self, k: str) -> str | None:
        """Returns 'quit', 'poll' or None."""
        if k in ("q", "Q", "\x03"):
            return "quit"
        if k in ("1", "2", "3", "4"):
            self.view, self.help = int(k) - 1, False
        elif k in ("\t", "right", "n"):
            self.idx += 1
        elif k in ("left", "p"):
            self.idx -= 1
        elif k == " ":
            self.frozen = not self.frozen
        elif k in ("t", "T"):
            self.theme = (self.theme + 1) % len(THEMES)
        elif k in ("?", "h"):
            self.help = not self.help
        elif k == "esc":
            self.help = False
        elif k in ("c", "C"):
            for s in self.servers.values():
                s.clear()
            self.hw.clear()
        elif k in ("r", "R"):
            return "poll"
        return None

    # ---- frame -----------------------------------------------------------
    def render(self):
        th, w = self.th, self.console.width
        s = self.cur
        parts = [self.r_header(s), self.r_tabs(), Text("─" * w, style=th["border"])]
        if self.help:
            parts.append(self.r_help())
        elif self.view == 3:
            parts.append(self.r_hardware_view(w))
        elif s is None:
            parts.append(self.r_empty())
        elif s.error:
            parts.append(self.r_state(s, w))
            parts.append(Panel(Text(s.error, style=th["bad"]), title=" server error ", title_align="left",
                               border_style=th["bad"], box=box.ROUNDED))
            parts.append(self.r_tiles(s, w))
        elif self.view == 0:
            parts += [self.r_state(s, w), self.r_tiles(s, w), self.r_bottom(s, w)]
        elif self.view == 1:
            parts += [self.r_state(s, w), self.r_slots_view(s, w)]
        else:
            parts += [self.r_state(s, w), self.r_requests_view(s)]
        parts.append(self.r_footer())
        return Group(*parts)

    def r_header(self, s: Server | None):
        th = self.th
        left = Text()
        left.append(" LLAMA ", style=f"bold {th['badge']}")
        if s is None:
            left.append("  Waiting", style=f"bold {th['text']}")
        else:
            i = s.info
            left.append("  Error" if s.error else "  Serving", style=f"bold {th['bad'] if s.error else th['text']}")
            left.append(f"   {s.base}", style=th["dim"])
            left.append(f"   {i['model']}", style=f"bold {th['accent']}")
            ctx = f" {fk(i['ctx'])}" if i.get("ctx") else ""
            par = f" ×{i['parallel']}" if i.get("parallel") else ""
            left.append(f"{ctx}{par}", style=th["dim"])
        right = Text()
        if s is not None and not s.error:
            st = s.state()
            pr = s.progress()
            if st in ("reading", "generating"):
                word = "Reading prompt" if st == "reading" else "Generating"
                col = th["pp"] if st == "reading" else th["gen"]
                right.append(f"● {word}", style=f"bold {col}")
                if pr:
                    right.append(f" · {pr[0] * 100:.0f}%", style=col)
                right.append("   ")
            elif st == "queued":
                right.append("● Queued   ", style=f"bold {th['warn']}")
        right.append("[FROZEN]" if self.frozen else "[LIVE]",
                     style=f"bold {th['warn']}" if self.frozen else f"bold {th['ok']}")
        g = Table.grid(expand=True)
        g.add_column(no_wrap=True, overflow="ellipsis")
        g.add_column(justify="right", no_wrap=True)
        g.add_row(left, right)
        return g

    def r_tabs(self):
        th = self.th
        t = Text(" ")
        for n, name in enumerate(VIEWS):
            if n == self.view and not self.help:
                t.append(f"[{n + 1} {name}]", style=f"bold {th['accent']}")
            else:
                t.append(f" {n + 1} {name} ", style=th["dim"])
            t.append("  ")
        right = Text()
        if len(self.order) > 1:
            right.append("◀ ", style=th["dim"])
            for n, port in enumerate(self.order):
                srv = self.servers[port]
                st = srv.state()
                dot = {"generating": th["gen"], "reading": th["pp"], "queued": th["warn"],
                       "error": th["bad"]}.get(st, th["dim"])
                right.append("● ", style=dot)
                right.append(srv.info["model"][:18], style=f"bold {th['accent']}" if n == self.idx else th["dim"])
                right.append("  ")
            right.append("▶ Tab", style=th["dim"])
        g = Table.grid(expand=True)
        g.add_column(no_wrap=True)
        g.add_column(justify="right", no_wrap=True, overflow="ellipsis")
        g.add_row(t, right)
        return g

    def r_state(self, s: Server, w: int):
        th = self.th
        st = s.state()
        t = Text(" Model state  ", style=th["dim"])
        for key, label, col in (("idle", "Idle", "grey62"), ("reading", "Reading", th["pp"]),
                                ("generating", "Generating", th["gen"]), ("queued", "Queued", th["warn"]),
                                ("error", "Error", th["bad"])):
            if key == st:
                t.append(f" {label} ", style=f"bold black on {col}")
            else:
                t.append(f" {label} ", style="grey35")
            t.append(" ")
        pr = s.progress()
        g = Table.grid(expand=True)
        g.add_column(no_wrap=True)
        g.add_column(justify="right", no_wrap=True)
        g.add_row(t, Text(pr[1] if pr else ("idle · waiting for requests" if st == "idle" else ""),
                          style=th["text"]))
        col = th["pp"] if st == "reading" else th["gen"]
        return Group(g, Text(" ") + bar(pr[0] if pr else 0, w - 2, col, th["empty"]), Text(""))

    # ---- tiles -----------------------------------------------------------
    def tile(self, title: str, value: str, sub: str, hist, top: float, color: str, width: int):
        th = self.th
        inner = max(4, width - 4)
        body = Group(Text(value, style=f"bold {color}", no_wrap=True, overflow="ellipsis"),
                     Text(sub, style=th["dim"], no_wrap=True, overflow="ellipsis"),
                     spark(hist, top, inner, color, th["empty"]))
        return Panel(body, title=Text(f" {title} ", style=th["dim"]), title_align="left", box=box.ROUNDED,
                     border_style=th["border"], padding=(0, 1), width=width)

    def r_tiles(self, s: Server, w: int):
        th, hw = self.th, self.hw
        with hw.lock:
            g = dict(hw.gpus[0]) if hw.gpus else {}
            h = {k: list(v) for k, v in hw.hist.items()}
            cpu, cores, threads = hw.cpu, hw.cores, hw.threads
            dr, dwr = hw.disk_r, hw.disk_w
        st = s.state()
        if st == "reading":
            speed, sub, hist, top, col = s.pp_rate, "reading the prompt", s.pp_hist, max(50, s.peak_pp), th["pp"]
        elif st in ("generating", "queued"):
            speed, sub, hist, top, col = (s.gen_rate, f"generating · peak {s.peak_gen:.1f}", s.gen_hist,
                                          max(20, s.peak_gen * 1.1), th["gen"])
        else:
            lg = s.last_gen
            speed = lg["tps"] if lg else 0.0
            sub = f"last request · {ago(lg['when'])}" if lg else "idle"
            hist, top, col = s.gen_hist, max(20, s.peak_gen * 1.1), th["gen"]
        tiles = [("Speed", f"{speed:.1f} tok/s", sub, hist, top, col)]
        if g:
            util = g.get("util") or 0
            tiles.append(("GPU load", f"{util:.0f} %", g.get("name", ""), h.get("g0.util", []), 100, th["gpu"]))
            used, total = g.get("used") or 0, g.get("total") or 1
            vf = used / total
            tiles.append(("VRAM", f"{used:.1f} / {total:.0f} GB", f"{vf * 100:.0f}% used · {total - used:.1f} GB free",
                          h.get("g0.vram", []), 100, level(vf, th["vram"], th, 0.92, 0.98)))
            temp = g.get("temp") or 0
            extra = []
            if g.get("fan") is not None:
                extra.append(f"fan {g['fan']:.0f}%")
            if g.get("sm"):
                extra.append(f"SM {g['sm']:.0f} MHz")
            tiles.append(("GPU temp", f"{temp:.0f} °C", " · ".join(extra), h.get("g0.temp", []), 100,
                          level(temp / 100, th["temp"], th, 0.78, 0.86)))
            pw, lim = g.get("power") or 0, g.get("limit") or 0
            tiles.append(("Power", f"{pw:.0f} W", f"of {lim:.0f} W limit" if lim else "", h.get("g0.power", []),
                          lim or max(h.get("g0.power") or [1]), th["power"]))
            gen, width, mg = g.get("gen"), g.get("width"), g.get("max_gen")
            bwmax = PCIE_LANE_GBPS.get(mg or gen or 4, 2) * (width or 16) * 1000  # MB/s
            link = f"Gen{gen} x{width}" if gen else "PCIe"
            if mg and gen and mg != gen:
                link += f"  (max {mg})"
            rx, tx = g.get("rx"), g.get("tx")
            psub = f"to GPU {rx:,.0f} · from {tx:,.0f} MB/s" if rx is not None else "throughput n/a"
            tiles.append(("PCIe", link, psub, h.get("g0.rx", []), bwmax, th["pcie"]))
        else:
            tiles.append(("GPU", "n/a", self.hw.gpu_err or "reading...", [], 1, th["dim"]))
        tiles.append(("CPU", f"{cpu:.0f} %", f"{cores} cores · {threads} threads", h.get("cpu", []), 100,
                      level(cpu / 100, th["cpu"], th, 0.9, 0.98)))
        tiles.append(("Disk read", f"{dr:,.1f} MB/s", f"write {dwr:,.1f} MB/s", h.get("disk_r", []),
                      max(h.get("disk_r") or [1]) or 1, th["disk"]))

        ncols = 4 if w >= 96 else 2 if w >= 48 else 1
        tw = (w - (ncols - 1)) // ncols
        grid = Table.grid(padding=(0, 1, 0, 0))
        for _ in range(ncols):
            grid.add_column(width=tw)
        row = []
        for t in tiles:
            row.append(self.tile(*t, width=tw))
            if len(row) == ncols:
                grid.add_row(*row)
                row = []
        if row:
            grid.add_row(*(row + [Text("")] * (ncols - len(row))))
        return grid

    # ---- overview bottom -------------------------------------------------
    def fill_rows(self, s: Server, width: int) -> list:
        th = self.th
        rows = []

        def add(label, frac, right, color):
            head = Table.grid(expand=True)
            head.add_column(no_wrap=True)
            head.add_column(justify="right", no_wrap=True)
            head.add_row(Text(f" {label}", style=th["text"]), Text(right, style=th["dim"]))
            rows.append(head)
            rows.append(Text(" ") + bar(frac, width - 1, color, th["empty"]))

        p = s.primary()
        n_ctx_cfg = int(s.info.get("ctx") or 0)
        if p:
            used = (p.get("n_prompt_tokens", 0) or 0) + (s.next_token(p).get("n_decoded", 0) or 0)
            n_ctx = p.get("n_ctx") or n_ctx_cfg
            frac = used / n_ctx if n_ctx else 0
            add("Context fill", frac, f"{fk(used)} / {fk(n_ctx)}  {frac * 100:.0f}%",
                level(frac, th["ok"], th, 0.8, 0.92))
        else:
            mx = s.metrics.get("n_tokens_max", 0)
            n_ctx = (s.slots[0].get("n_ctx") if s.slots else None) or n_ctx_cfg
            add("Context fill", 0, f"idle · max seen {fk(mx)} / {fk(n_ctx)}", th["ok"])
        m = s.metrics
        draft, acc, drafts = (m.get("spec_decode_num_draft_tokens_total", 0),
                              m.get("spec_decode_num_accepted_tokens_total", 0),
                              m.get("spec_decode_num_drafts_total", 0))
        if draft:
            per = m.get("spec_decode_num_accepted_tokens_per_pos_total") or {}
            pos = "/".join(f"{per[k] / drafts * 100:.0f}" for k in sorted(per)) if drafts else ""
            add("Draft acceptance", acc / draft,
                f"{acc / draft * 100:.0f}% · {acc / drafts:.2f}/draft" + (f" · {pos}" if pos else ""),
                th["draft"])
        busy, n = len(s.busy_slots()), s.n_slots()
        dfr = m.get("requests_deferred", 0)
        add("Slots busy", busy / n, f"{busy} / {n}" + (f" · {dfr:.0f} queued" if dfr else ""), th["slots"])
        with self.hw.lock:
            used, total = self.hw.ram
        rss = s.rss()
        add("System RAM", used / total, f"{used:.1f} / {total:.0f} GB" + (f" · server {rss / GB:.1f}" if rss else ""),
            level(used / total, th["ram"], th, 0.9, 0.97))
        return rows

    def requests_table(self, s: Server, limit: int):
        th = self.th
        t = Table(box=None, pad_edge=False, show_edge=False, header_style=f"bold {th['dim']}", padding=(0, 1))
        for col, j in (("TIME", "left"), ("STATUS", "left"), ("PROMPT", "right"), ("REUSED", "right"),
                       ("OUTPUT", "right"), ("TOK/S", "right"), ("DURATION", "right")):
            t.add_column(col, justify=j, no_wrap=True)
        n = 0
        for r in sorted(s.active.values(), key=lambda r: r["start"], reverse=True):
            if n >= limit:
                break
            gen = bool(r.get("out"))
            rate = s.slot_rate.get(r["slot"])
            t.add_row(datetime.fromtimestamp(r["start"]).strftime("%H:%M:%S"),
                      Text("Gen" if gen else "Read", style=f"bold {th['gen'] if gen else th['pp']}"),
                      fint(r.get("prompt")), fint(r.get("reused")), fint(r.get("out")),
                      f"{rate:.1f}" if rate and gen else "-", f"{time.time() - r['start']:.1f} s")
            n += 1
        for r in s.requests:
            if n >= limit:
                break
            t.add_row(datetime.fromtimestamp(r["time"]).strftime("%H:%M:%S"),
                      Text("Done", style=th["ok"]), fint(r["prompt"]), fint(r["reused"]), fint(r["out"]),
                      f"{r['tps']:.1f}" if r.get("tps") else "-", f"{r['dur']:.1f} s")
            n += 1
        if n == 0:
            return Group(t, Text("no requests yet - they appear here as they run", style=th["dim"]))
        return t

    def r_bottom(self, s: Server, w: int):
        th = self.th
        if w < 96:
            return Group(*self.fill_rows(s, w - 1), Text(""), Text(" Recent requests", style=f"bold {th['text']}"),
                         self.requests_table(s, 4))
        lw = max(30, int(w * 0.42))
        rows = self.fill_rows(s, lw)
        right = Group(Text("Recent requests", style=f"bold {th['text']}"),
                      self.requests_table(s, max(3, len(rows) - 2)))
        g = Table.grid(padding=(0, 3))
        g.add_column(width=lw)
        g.add_column()
        g.add_row(Group(*rows), right)
        return g

    # ---- view 2: slots ---------------------------------------------------
    def r_slots_view(self, s: Server, w: int):
        th = self.th
        if s.slots_error:
            return Text(f" {s.slots_error} - the Slots view needs the /slots endpoint", style=th["dim"])
        cw = max(6, min(40, w - 114))
        t = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style=f"bold {th['dim']}", border_style=th["border"],
                  expand=True)
        for col, kw in (("SLOT", dict(justify="right")), ("STATE", {}), ("TASK", dict(justify="right")),
                        ("PROMPT", dict(justify="right")), ("REUSED", dict(justify="right")),
                        ("READ", dict(justify="right")), ("OUTPUT", dict(justify="right")),
                        ("LEFT", dict(justify="right")), ("TOK/S", dict(justify="right")),
                        ("SAMPLING", {}), ("CONTEXT", dict(min_width=cw + 12))):
            kw.setdefault("min_width", len(col))
            t.add_column(col, no_wrap=True, **kw)
        for sl in s.slots:
            n_ctx = sl.get("n_ctx") or 0
            if not sl.get("is_processing"):
                t.add_row(str(sl.get("id")), Text("idle", style=th["dim"]), "-", "-", "-", "-", "-", "-", "-", "",
                          bar(0, cw, th["ok"], th["empty"]) + Text(f" 0 / {fk(n_ctx)}", style=th["dim"]))
                continue
            nt = s.next_token(sl)
            dec = nt.get("n_decoded", 0) or 0
            remain = nt.get("n_remain")
            n_prompt = sl.get("n_prompt_tokens", 0) or 0
            cache = sl.get("n_prompt_tokens_cache", 0) or 0
            done = sl.get("n_prompt_tokens_processed", 0) or 0
            state = Text("gen", style=f"bold {th['gen']}") if dec else Text("read", style=f"bold {th['pp']}")
            prm = sl.get("params") or {}
            samp = []
            for k, lab in (("temperature", "T"), ("top_p", "p"), ("top_k", "k")):
                v = prm.get(k)
                if isinstance(v, (int, float)):
                    samp.append(f"{lab}{v:g}")
            used = n_prompt + dec
            frac = used / n_ctx if n_ctx else 0
            ctx = bar(frac, cw, level(frac, th["ok"], th, 0.8, 0.92), th["empty"])
            ctx.append(f" {fk(used)} / {fk(n_ctx)}", style=th["dim"])
            rate = s.slot_rate.get(sl.get("id"))
            t.add_row(str(sl.get("id")), state, str(sl.get("id_task", "-")), fint(n_prompt), fint(cache),
                      fint(done), fint(dec), "∞" if remain is None or remain < 0 else fint(remain),
                      f"{rate:.1f}" if rate and dec else "-", Text(" ".join(samp), style=th["dim"]), ctx)
        smap = Text(" Slot map  ", style=th["dim"])
        for sl in s.slots:
            if sl.get("is_processing"):
                smap.append("■ ", style=th["gen"] if s.next_token(sl).get("n_decoded") else th["pp"])
            else:
                smap.append("□ ", style="grey35")
        m = s.metrics
        smap.append(f"   processing {m.get('requests_processing', 0):.0f} · queued {m.get('requests_deferred', 0):.0f}"
                    f" · busy slots per decode {m.get('n_busy_slots_per_decode', 0):.2f}", style=th["dim"])
        return Group(smap, t)

    # ---- view 3: requests ------------------------------------------------
    def r_requests_view(self, s: Server):
        th = self.th
        done = list(s.requests)
        rated = [r for r in done if r.get("tps")]
        m = s.metrics
        tot_p, tot_c = m.get("prompt_tokens_total", 0), m.get("prompt_tokens_cached_total", 0)
        summary = Table.grid(padding=(0, 4))
        for _ in range(6):
            summary.add_column()

        def stat(label, value):
            return Group(Text(label, style=th["dim"]), Text(value, style=f"bold {th['accent']}"))

        summary.add_row(
            stat(" requests seen", " " + fint(len(done))),
            stat("avg gen tok/s", f"{sum(r['tps'] for r in rated) / len(rated):.1f}" if rated else "-"),
            stat("lifetime output", fint(m.get("tokens_predicted_total", 0))),
            stat("lifetime prompt", fint(tot_p)),
            stat("prompt cache hit", f"{tot_c / (tot_p + tot_c) * 100:.0f}%" if tot_p + tot_c else "-"),
            stat("server avg gen", f"{m.get('tokens_predicted_total', 0) / m['tokens_predicted_seconds_total']:.1f} t/s"
                 if m.get("tokens_predicted_seconds_total") else "-"),
        )
        limit = max(5, self.console.height - 15)
        return Group(summary, Text(""), self.requests_table(s, limit), Text(""),
                     Text(" history is collected while the dashboard runs · OUTPUT / TOK/S are exact when one request "
                          "finishes per poll", style=th["dim"]))

    # ---- view 4: hardware ------------------------------------------------
    def r_hardware_view(self, w: int):
        th, hw = self.th, self.hw
        with hw.lock:
            gpus = [dict(g) for g in hw.gpus]
            h = {k: list(v) for k, v in hw.hist.items()}
            percpu = list(hw.percpu)
            cpu, (ru, rt), dr, dwr = hw.cpu, hw.ram, hw.disk_r, hw.disk_w
        bw = max(10, min(30, w // 4))
        sw = max(10, w - bw - 14 - 24 - 4)
        g = Table.grid(padding=(0, 1))
        g.add_column(width=14, no_wrap=True, style=f"bold {th['text']}")
        g.add_column(width=bw, no_wrap=True)
        g.add_column(width=24, no_wrap=True, justify="right")
        g.add_column(no_wrap=True)

        def row(label, frac, value, hist, top, color):
            g.add_row(f" {label}", bar(frac, bw, color, th["empty"]), Text(value, style=color),
                      spark(hist, top, sw, color, th["empty"]))

        for i, gp in enumerate(gpus):
            p = f"g{i}."
            tag = f"GPU{i} " if len(gpus) > 1 else ""
            g.add_row(Text(f"{tag}{gp['name']}", style=f"bold {th['accent']}"), "", "", "")
            u = gp.get("util") or 0
            row("load", u / 100, f"{u:.0f} %", h.get(p + "util", []), 100, th["gpu"])
            if gp.get("used") is not None:
                vf = gp["used"] / gp["total"]
                row("VRAM", vf, f"{gp['used']:.1f} / {gp['total']:.1f} GB", h.get(p + "vram", []), 100,
                    level(vf, th["vram"], th, 0.92, 0.98))
            t = gp.get("temp") or 0
            row("temp", t / 100, f"{t:.0f} °C", h.get(p + "temp", []), 100,
                level(t / 100, th["temp"], th, 0.78, 0.86))
            pw, lim = gp.get("power") or 0, gp.get("limit") or 1
            row("power", pw / lim, f"{pw:.0f} / {lim:.0f} W", h.get(p + "power", []), lim, th["power"])
            if gp.get("sm"):
                smh = h.get(p + "sm") or [1]
                row("SM clock", gp["sm"] / max(max(smh), 1), f"{gp['sm']:.0f} MHz"
                    + (f" · mem {gp['memclk']:.0f}" if gp.get("memclk") else ""), smh, max(smh), th["gpu"])
            if gp.get("fan") is not None:
                row("fan", gp["fan"] / 100, f"{gp['fan']:.0f} %", h.get(p + "fan", []), 100, th["dim"])
            if gp.get("rx") is not None:
                bwmax = PCIE_LANE_GBPS.get(gp.get("max_gen") or gp.get("gen") or 4, 2) * (gp.get("width") or 16) * 1000
                row(f"PCIe ↓ G{gp.get('gen')}x{gp.get('width')}", gp["rx"] / bwmax, f"{gp['rx']:,.0f} MB/s to GPU",
                    h.get(p + "rx", []), bwmax, th["pcie"])
                row("PCIe ↑", gp["tx"] / bwmax, f"{gp['tx']:,.0f} MB/s from GPU", h.get(p + "tx", []), bwmax,
                    th["pcie"])
            g.add_row("", "", "", "")
        if not gpus:
            g.add_row(" GPU", Text(hw.gpu_err or "reading...", style=th["dim"]), "", "")
        g.add_row(Text("System", style=f"bold {th['accent']}"), "", "", "")
        row("CPU", cpu / 100, f"{cpu:.0f} % · {hw.cores}c/{hw.threads}t", h.get("cpu", []), 100, th["cpu"])
        cores = Text()
        for v in percpu:
            cores.append(SPARK[min(7, int(v / 100 * 7.999))] if v > 0 else "▁",
                         style=level(v / 100, th["cpu"], th, 0.9, 0.98) if v > 0 else th["empty"])
        g.add_row(" per thread", cores, Text(f"{len(percpu)} logical", style=th["dim"]), "")
        row("RAM", ru / rt, f"{ru:.1f} / {rt:.0f} GB", h.get("ram", []), 100,
            level(ru / rt, th["ram"], th, 0.9, 0.97))
        dtop = max((h.get("disk_r") or [1]) + (h.get("disk_w") or [1]) + [1])
        row("disk read", dr / dtop, f"{dr:,.1f} MB/s", h.get("disk_r", []), dtop, th["disk"])
        row("disk write", dwr / dtop, f"{dwr:,.1f} MB/s", h.get("disk_w", []), dtop, th["disk"])
        return Group(g, Text(f" sparklines: last {sw} samples every {hw.period:g}s", style=th["dim"]))

    # ---- misc ------------------------------------------------------------
    def r_help(self):
        th = self.th
        t = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
        t.add_column(style=f"bold {th['accent']}", no_wrap=True)
        t.add_column(style=th["text"])
        for k, d in (("1-4", "switch view: Overview · Slots · Requests · Hardware"),
                     ("Tab  ← →", "switch between loaded models (one llama-server per model)"),
                     ("Space", "freeze / unfreeze polling"), ("r", "poll now"),
                     ("t", f"cycle theme ({', '.join(n for n, _ in THEMES)})"),
                     ("c", "clear history, peaks and request log"), ("? / Esc", "toggle this help"),
                     ("q", "quit (servers keep running)"), ("", ""),
                     ("Speed", "live tok/s from /slots n_decoded deltas (EMA); idle shows the last request"),
                     ("Draft", "MTP / speculative acceptance = accepted / drafted tokens; per-position %"),
                     ("Requests", "built from slot task transitions; exact numbers come from /metrics"),
                     ("Flags", "each model needs  --metrics --api-key-file <file>  in its custom flags")):
            t.add_row(k, d)
        return Panel(t, title=" help ", title_align="left", border_style=th["accent"], box=box.ROUNDED)

    def r_empty(self):
        return Panel(Text.from_markup(
            "No llama-server process found.\n\n"
            "Load a model in LM Studio with custom flags:\n"
            "  [bold]--metrics --api-key-file <path-to>/metrics-key.txt[/]\n"
            "or start your own llama-server with [bold]--metrics[/].\n\n"
            "The Hardware view (4) works without a server."),
            border_style=self.th["warn"], title=" waiting ", title_align="left", box=box.ROUNDED)

    def r_footer(self):
        th = self.th
        keys = Text()
        for k, d in (("1-4", "views"), ("Tab", "model"), ("Space", "freeze"), ("r", "refresh"), ("t", "theme"),
                     ("c", "clear"), ("?", "help"), ("q", "quit")):
            keys.append(f" {k}", style=f"bold {th['accent']}")
            keys.append(f" {d}  ", style=th["dim"])
        s = self.cur
        status = (f" Read only  |  {len(self.servers)} server(s)  |  poll {self.args.interval:g}s"
                  f"  |  theme {THEMES[self.theme][0]}")
        if s and s.info.get("runtime"):
            status += f"  |  {s.info['runtime']}"
        if s and s.info.get("pid"):
            status += f"  |  pid {s.info['pid']}"
        return Group(Text(""), keys, Text(status, style=th["dim"], no_wrap=True, overflow="ellipsis"))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive terminal dashboard for llama-server / LM Studio.")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="poll interval in seconds (default 1)")
    ap.add_argument("-p", "--port", type=int, action="append", help="only watch this port (repeatable)")
    ap.add_argument("-k", "--key-file", help="API key file (overrides the server's --api-key-file)")
    ap.add_argument("--view", type=int, default=1, help="start view 1-4 (default 1 Overview)")
    ap.add_argument("--theme", default=None, help="start theme: " + ", ".join(n for n, _ in THEMES))
    ap.add_argument("--no-screen", action="store_true", help="draw inline instead of the full-screen buffer")
    ap.add_argument("--no-gpu", action="store_true", help="don't read GPU sensors")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args()

    console = Console()
    app = App(args, console)
    if args.theme:
        names = [n for n, _ in THEMES]
        if args.theme in names:
            app.theme = names.index(args.theme)

    if args.once:
        app.tick()
        deadline = time.monotonic() + max(args.interval, app.hw.period) * 2 + 1
        while app.hw.samples < 2 and time.monotonic() < deadline:
            time.sleep(0.1)
        app.tick()
        console.print(app.render())
        return

    keys = Keys()
    try:
        with Live(app.render(), console=console, screen=not args.no_screen, auto_refresh=False,
                  transient=False) as live:
            next_poll = 0.0
            while True:
                changed = False
                for k in keys.read():
                    act = app.handle(k)
                    if act == "quit":
                        return
                    if act == "poll":
                        next_poll = 0.0
                    changed = True
                now = time.monotonic()
                if now >= next_poll:
                    if not app.frozen:
                        app.tick()
                    next_poll = now + args.interval
                    changed = True
                if changed:
                    live.update(app.render(), refresh=True)
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()


if __name__ == "__main__":
    main()
