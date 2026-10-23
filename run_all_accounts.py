#!/usr/bin/env python3
"""
run_all_accounts.py
====================
Launches one full API instance (LiveTrader + its own MT5 connection) per
account listed in accounts.json — see accounts.example.json for the format.

Why N separate processes instead of one process handling several accounts:
the MetaTrader5 Python package only supports a single connection per OS
process (mt5.order_send / mt5.symbol_info etc. are module-global, not tied
to an instance). Each account therefore needs its own process, its own port,
and — on the MT5 side — its own terminal64.exe installation folder (a broker
login session can't be shared across two simultaneous different accounts,
even on the same server).

No changes to config/settings.py or any trading-logic file were needed for
this: each child process just receives different environment variables
(API_PORT, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, ...) before it
starts — exactly as if it had been launched alone with a different .env.

Usage
-----
    python run_all_accounts.py                  # launch all accounts, don't trade
    python run_all_accounts.py --start-trading   # also POST /trading/start on each
                                                  # once its API is healthy

Stop with Ctrl+C — every child is terminated cleanly (no orphaned MT5
terminal connections left behind).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
ACCOUNTS_PATH = PROJECT_ROOT / "accounts.json"
EXAMPLE_PATH = PROJECT_ROOT / "accounts.example.json"

# Fields from accounts.json that map directly to environment variable
# overrides for the child process. mt5_path/server/login/password map to
# the SAME env vars config/settings.py already reads from .env.
_ENV_FIELD_MAP = {
    "api_port":         "API_PORT",
    "mt5_login":        "MT5_LOGIN",
    "mt5_password":     "MT5_PASSWORD",
    "mt5_server":       "MT5_SERVER",
    "mt5_path":         "MT5_PATH",
    "initial_balance":  "TRADING_INITIAL_BALANCE",
}


def load_accounts() -> list[dict]:
    if not ACCOUNTS_PATH.exists():
        print(f"No accounts.json found at {ACCOUNTS_PATH}")
        print(f"Copy {EXAMPLE_PATH.name} to accounts.json and fill in real credentials first.")
        raise SystemExit(1)
    data = json.loads(ACCOUNTS_PATH.read_text(encoding="utf-8"))
    accounts = data.get("accounts", [])
    if not accounts:
        print("accounts.json has no accounts listed.")
        raise SystemExit(1)
    ports = [a["api_port"] for a in accounts]
    if len(ports) != len(set(ports)):
        print(f"Duplicate api_port values in accounts.json: {ports}")
        raise SystemExit(1)
    return accounts


def build_child_env(account: dict) -> dict:
    env = os.environ.copy()
    for field, env_var in _ENV_FIELD_MAP.items():
        if field in account and account[field] not in (None, ""):
            env[env_var] = str(account[field])
    return env


class ChildProcess:
    def __init__(self, account: dict) -> None:
        self.account = account
        self.label   = account.get("label", account.get("id", "account"))
        self.port    = account["api_port"]
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        env = build_child_env(self.account)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "api.main"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump_output, daemon=True).start()

    def _pump_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        prefix = f"[{self.label}:{self.port}]"
        for line in self.process.stdout:
            print(f"{prefix} {line.rstrip()}")

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait_healthy(self, timeout_s: float = 30.0) -> bool:
        import requests
        deadline = time.monotonic() + timeout_s
        url = f"http://localhost:{self.port}/health"
        while time.monotonic() < deadline:
            if not self.is_alive():
                return False
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False

    def start_trading(self) -> bool:
        import requests
        body = {}
        if "initial_balance" in self.account:
            body["initial_balance"] = self.account["initial_balance"]
        if self.account.get("ftmo_enabled"):
            body["ftmo_enabled"] = True
            body["ftmo_initial_balance"] = self.account.get(
                "ftmo_initial_balance", self.account.get("initial_balance", 0),
            )
        try:
            r = requests.post(
                f"http://localhost:{self.port}/trading/start",
                json=body, timeout=15,
            )
            ok = r.status_code == 200
            print(f"[{self.label}:{self.port}] /trading/start -> {r.status_code} {r.text[:200]}")
            return ok
        except Exception as exc:
            print(f"[{self.label}:{self.port}] /trading/start failed: {exc}")
            return False

    def stop(self) -> None:
        if self.process is None or not self.is_alive():
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[{self.label}:{self.port}] did not exit in time — killing.")
            self.process.kill()
            self.process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-trading", action="store_true",
        help="Also call POST /trading/start on every account once its API is healthy "
             "(otherwise the servers just come up idle, matching how a single-account "
             "instance behaves until you click Start).",
    )
    args = parser.parse_args()

    accounts = load_accounts()
    print(f"Launching {len(accounts)} account(s):")
    for a in accounts:
        print(f"  - {a.get('label', a.get('id'))} -> port {a['api_port']}")

    children = [ChildProcess(a) for a in accounts]
    try:
        for child in children:
            child.start()
            print(f"[{child.label}:{child.port}] started (pid={child.process.pid})")

        print("\nWaiting for every account's API to become healthy...")
        for child in children:
            if child.wait_healthy():
                print(f"[{child.label}:{child.port}] healthy.")
            else:
                print(f"[{child.label}:{child.port}] FAILED to become healthy "
                      f"(check its logs above — usually a bad MT5 path/login).")

        if args.start_trading:
            print("\nStarting trading on every healthy account...")
            for child in children:
                if child.is_alive():
                    child.start_trading()

        print("\nAll accounts launched. Press Ctrl+C to stop everything.\n")
        while any(c.is_alive() for c in children):
            time.sleep(1)
        print("All child processes have exited.")

    except KeyboardInterrupt:
        print("\nStopping all accounts...")
    finally:
        for child in children:
            child.stop()
        print("Done.")


if __name__ == "__main__":
    main()
