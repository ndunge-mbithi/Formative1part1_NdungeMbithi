"""Shared fixtures and the grouped stage report.

Fixtures for the test suites, plus a summary that tallies results by the
``group(letter, title)`` marker on each test and buckets them by the module's
``stage(number, title, target, slug=...)`` marker. Prints one block per stage
and writes ``stage<number>[_<slug>]_report.json``
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent

# nodeid -> {"stage": (number, title, target, slug), "group": (letter, title)}
_meta: dict[str, dict] = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "group(letter, title): test category within a stage, e.g. "
        "group('A', 'Output shape')",
    )
    config.addinivalue_line(
        "markers",
        "stage(number, title, target, slug=None): the assignment stage a test "
        "module covers",
    )


def _stage_key(item):
    marker = item.get_closest_marker("stage")
    if marker is None:
        return (None, "Stage report", "", None)
    args = list(marker.args)
    number = args[0] if args else marker.kwargs.get("number")
    title = args[1] if len(args) > 1 else marker.kwargs.get("title", "")
    target = args[2] if len(args) > 2 else marker.kwargs.get("target", "")
    slug = marker.kwargs.get("slug")
    return (number, title, target, slug)


def pytest_collection_modifyitems(config, items):
    for item in items:
        group_marker = item.get_closest_marker("group")
        if group_marker is None:
            continue
        letter = group_marker.args[0]
        if len(group_marker.args) > 1:
            gtitle = group_marker.args[1]
        else:
            gtitle = group_marker.kwargs.get("title", letter)
        _meta[item.nodeid] = {"stage": _stage_key(item), "group": (letter, gtitle)}


@pytest.fixture
def make_linear():
    """Return a factory ``make_linear(in, out, W=None, b=None) -> Linear``.

    When ``W`` / ``b`` are given they overwrite the freshly initialised
    parameters, so a test can assert ``forward`` against known values.
    """
    from nn.layers.linear import Linear

    def _make(in_features, out_features, W=None, b=None):
        layer = Linear(in_features, out_features)
        if W is not None:
            layer.W = np.asarray(W, dtype=float)
        if b is not None:
            layer.b = np.asarray(b, dtype=float)
        return layer

    return _make


@pytest.fixture
def make_softmax():
    """Return a factory ``make_softmax() -> Softmax`` (a fresh instance)."""
    from nn.activations.softmax import Softmax

    def _make():
        return Softmax()

    return _make


@pytest.fixture
def rng():
    """Deterministic RNG so randomised checks reproduce run to run."""
    return np.random.default_rng(20260910)


@pytest.fixture
def numeric_grad():
    """Return ``grad(loss_fn, param, eps=1e-6)`` -- central-difference gradient.

    ``loss_fn`` is a no-argument callable returning a scalar; ``param`` is the
    array to differentiate against and is perturbed in place and restored.
    """

    def _grad(loss_fn, param, eps=1e-6):
        out = np.zeros_like(param, dtype=float)
        for idx in np.ndindex(param.shape):
            orig = param[idx]
            param[idx] = orig + eps
            hi = loss_fn()
            param[idx] = orig - eps
            lo = loss_fn()
            param[idx] = orig
            out[idx] = (hi - lo) / (2 * eps)
        return out

    return _grad


def _find_ruff():
    local = Path(sys.executable).parent / "ruff"
    if local.exists():
        return str(local)
    return shutil.which("ruff")


def _lint_status():
    ruff = _find_ruff()
    if ruff is None or not (ROOT / "nn").is_dir():
        return {"tool": "ruff", "ran": False, "clean": None, "output": ""}
    proc = subprocess.run(
        [ruff, "check", "nn/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "tool": "ruff",
        "ran": True,
        "clean": proc.returncode == 0,
        "output": (proc.stdout or proc.stderr).strip()[-4000:],
    }


def _row(label, value):
    return f"  {label:<34}{value}"


def _report_filename(number, slug):
    if number is None:
        return "stage_report.json"
    if slug:
        return f"stage{number}_{slug}_report.json"
    return f"stage{number}_report.json"


def _collect_outcomes(terminalreporter):
    outcome: dict[str, str] = {}
    message: dict[str, str] = {}
    for key in ("passed", "failed", "error"):
        for rep in terminalreporter.stats.get(key, []):
            nodeid = getattr(rep, "nodeid", None)
            if nodeid is None or nodeid not in _meta:
                continue
            when = getattr(rep, "when", "call")
            if when == "call":
                outcome[nodeid] = rep.outcome
            elif when in ("setup", "teardown") and rep.outcome != "passed":
                outcome.setdefault(nodeid, "error")
            longrepr = getattr(rep, "longrepr", None)
            crash = getattr(longrepr, "reprcrash", None)
            if rep.outcome != "passed" and crash is not None:
                message[nodeid] = crash.message.splitlines()[0][:200]
    return outcome, message


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    tr = terminalreporter
    outcome, message = _collect_outcomes(tr)
    if not outcome:
        return

    buckets: dict[tuple, list[str]] = {}
    for nodeid in outcome:
        buckets.setdefault(_meta[nodeid]["stage"], []).append(nodeid)

    lint = _lint_status()
    width = 64

    for stage_key in sorted(
        buckets, key=lambda s: (s[0] is None, s[0] or 0, s[1] or "")
    ):
        number, title, target, slug = stage_key
        node_ids = buckets[stage_key]

        tally: dict[str, list[int]] = {}
        titles: dict[str, str] = {}
        failures = []
        for nodeid in node_ids:
            letter, gtitle = _meta[nodeid]["group"]
            titles.setdefault(letter, gtitle)
            row = tally.setdefault(letter, [0, 0])
            row[1] += 1
            if outcome[nodeid] == "passed":
                row[0] += 1
            else:
                failures.append(
                    (letter, nodeid, message.get(nodeid, outcome[nodeid]))
                )

        total_pass = sum(p for p, _ in tally.values())
        total_all = sum(t for _, t in tally.values())
        verdict = "PASS" if total_pass == total_all else "BLOCKED"

        head = f"STAGE {number}" if number is not None else "STAGE"
        out = ["", "=" * width, f"  {head}  --  {title}".rstrip()]
        if target:
            out.append(f"  {target}")
        out.append("=" * width)
        for letter in sorted(tally):
            p, t = tally[letter]
            status = "PASS" if p == t else "FAIL"
            out.append(_row(f"{letter}  {titles[letter]}", f"{p}/{t}   {status}"))
        out.append("-" * width)
        out.append(_row("TOTAL", f"{total_pass}/{total_all}"))
        if not lint["ran"]:
            lint_txt = "not run"
        else:
            lint_txt = "clean" if lint["clean"] else "ISSUES"
        out.append(_row("LINT (ruff check nn/)", lint_txt))
        if failures:
            out.append("-" * width)
            out.append("  FAILURES")
            for letter, nodeid, msg in failures:
                out.append(f"   {letter}  {nodeid.split('::', 1)[-1]}")
                out.append(f"      {msg}")
        out.append("-" * width)
        if verdict == "PASS":
            tail = "" if number is None else f" Stage {number}"
            out.append(f"  VERDICT:  PASS  --  all{tail} checks pass")
        else:
            k = len(failures)
            out.append(
                f"  VERDICT:  BLOCKED  --  fix the {k} failure"
                f"{'' if k == 1 else 's'} above"
            )
        out.append("=" * width)
        for line in out:
            tr.write_line(line)

        report = {
            "stage": number,
            "title": title,
            "target": target,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "categories": {
                _category_key(letter, titles[letter]): {
                    "passed": tally[letter][0],
                    "total": tally[letter][1],
                }
                for letter in sorted(tally)
            },
            "total": {"passed": total_pass, "total": total_all},
            "lint": lint,
            "verdict": verdict,
            "failures": [
                {"category": letter, "test": nid.split("::", 1)[-1], "message": m}
                for letter, nid, m in failures
            ],
        }
        fname = _report_filename(number, slug)
        (ROOT / fname).write_text(json.dumps(report, indent=2))
        tr.write_line(f"  wrote {ROOT / fname}")


def _category_key(letter, title):
    words = title.split()
    first = words[0].lower() if words else letter.lower()
    return f"{letter}_{first}"
