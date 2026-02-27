"""
Shared assertion + reporting for Docker E2E tests.

Used by: run_e2e.py, run_install.py, run_install_degraded.py, run_upgrade.py, run_ux.py

Usage:
    h = Harness("e2e")
    h.phase("Phase 1: Filesystem")
    h.check("registry exists", path.exists(), "missing")
    h.skip("MCP test", "no credentials")
    sys.exit(h.finish())

JSON output: /tmp/flex-test-{suite}.json
"""
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    phase: str = ""


@dataclass
class SuiteResult:
    suite: str
    started_at: str = ""
    elapsed_s: float = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    checks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "started_at": self.started_at,
            "elapsed_s": round(self.elapsed_s, 2),
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "checks": [asdict(c) for c in self.checks],
        }


class Harness:
    """Accumulates check results, prints live, writes JSON on exit."""

    def __init__(self, suite_name: str, json_dir: Optional[str] = None):
        self.suite = suite_name
        self._json_dir = Path(json_dir) if json_dir else Path("/tmp")
        self.result = SuiteResult(suite=suite_name)
        self.result.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._start = time.time()
        self._current_phase = ""

    def phase(self, name: str):
        """Start a named phase (prints header)."""
        self._current_phase = name
        print()
        print("=" * 60)
        print(name)
        print("=" * 60)

    def check(self, name: str, condition: bool, detail: str = ""):
        """Assert a condition. Prints live, records result."""
        cr = CheckResult(
            name=name,
            passed=bool(condition),
            detail=detail,
            phase=self._current_phase,
        )
        self.result.checks.append(cr)
        if condition:
            self.result.passed += 1
            print(f"  {PASS} {name}")
        else:
            self.result.failed += 1
            msg = f"{name}" + (f": {detail}" if detail else "")
            print(f"  {FAIL} {msg}")

    def skip(self, name: str, reason: str = ""):
        """Record a skipped check."""
        cr = CheckResult(name=name, passed=True, detail=reason, phase=self._current_phase)
        self.result.checks.append(cr)
        self.result.skipped += 1
        print(f"  {SKIP} {name}" + (f": {reason}" if reason else ""))

    def finish(self) -> int:
        """Print summary, write JSON, return exit code (0=pass, 1=fail)."""
        self.result.elapsed_s = time.time() - self._start

        # Write JSON
        json_path = self._json_dir / f"flex-test-{self.suite}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.result.to_dict(), indent=2))

        # Terminal summary
        print()
        print("=" * 60)
        if self.result.failed:
            print(f"\033[31mFAILED \u2014 {self.result.failed} checks failed:\033[0m")
            for c in self.result.checks:
                if not c.passed:
                    d = f": {c.detail}" if c.detail else ""
                    print(f"  \u2022 {c.name}{d}")
        else:
            skipped = f", {self.result.skipped} skipped" if self.result.skipped else ""
            print(f"\033[32m{self.result.passed} checks passed{skipped} \u2014 "
                  f"{self.suite} OK ({self.result.elapsed_s:.1f}s)\033[0m")

        print(f"  JSON: {json_path}")
        return 1 if self.result.failed else 0
