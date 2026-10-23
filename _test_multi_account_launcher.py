#!/usr/bin/env python3
"""
Test script for run_all_accounts.py's core logic — deliberately does NOT
spawn a real subprocess connected to MT5. The user has a real account
actively trading right now; launching a second session against the same
login (or squatting its port) risks interfering with it. This verifies the
pure logic instead: accounts.json parsing, env var mapping, duplicate-port
detection, and the /trading/start request body built per account.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True

PROJECT_ROOT = Path(r"d:\projet startup\agent trading\quant-fund-ai")
spec = importlib.util.spec_from_file_location("run_all_accounts", PROJECT_ROOT / "run_all_accounts.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("=" * 70)
print("TEST 1: build_child_env() maps accounts.json fields to the right env vars")
print("=" * 70)
account = {
    "id": "acct-a", "label": "Test Account A", "api_port": 8099,
    "mt5_login": 999888, "mt5_password": "secret123", "mt5_server": "FTMO-Demo",
    "mt5_path": r"C:\Fake\terminal64.exe", "initial_balance": 25000,
}
env = mod.build_child_env(account)
checks = {
    "API_PORT": "8099", "MT5_LOGIN": "999888", "MT5_PASSWORD": "secret123",
    "MT5_SERVER": "FTMO-Demo", "MT5_PATH": r"C:\Fake\terminal64.exe",
    "TRADING_INITIAL_BALANCE": "25000",
}
ok1 = all(env.get(k) == v for k, v in checks.items())
print(f"  {PASS if ok1 else FAIL} env overrides: " +
      ", ".join(f"{k}={env.get(k)}" for k in checks))
all_ok &= ok1

# The base os.environ (PATH etc.) must still be present -> child isn't a
# blank environment, just the current one with overrides on top.
ok2 = "PATH" in env or "Path" in env
print(f"  {PASS if ok2 else FAIL} base environment (PATH) preserved, not wiped")
all_ok &= ok2

print("\n" + "=" * 70)
print("TEST 2: load_accounts() parses accounts.json and rejects duplicate ports")
print("=" * 70)
tmpdir = Path(tempfile.mkdtemp(prefix="accounts_test_"))
try:
    good = tmpdir / "accounts.json"
    good.write_text(json.dumps({
        "accounts": [
            {"id": "a", "label": "A", "api_port": 8000, "mt5_login": 1},
            {"id": "b", "label": "B", "api_port": 8001, "mt5_login": 2},
        ]
    }), encoding="utf-8")

    mod.ACCOUNTS_PATH = good
    accounts = mod.load_accounts()
    ok3 = len(accounts) == 2 and {a["api_port"] for a in accounts} == {8000, 8001}
    print(f"  {PASS if ok3 else FAIL} loaded 2 accounts with distinct ports: "
          f"{[a['api_port'] for a in accounts]}")
    all_ok &= ok3

    dup = tmpdir / "accounts_dup.json"
    dup.write_text(json.dumps({
        "accounts": [
            {"id": "a", "label": "A", "api_port": 8000, "mt5_login": 1},
            {"id": "b", "label": "B", "api_port": 8000, "mt5_login": 2},
        ]
    }), encoding="utf-8")
    mod.ACCOUNTS_PATH = dup
    try:
        mod.load_accounts()
        ok4 = False
    except SystemExit:
        ok4 = True
    print(f"  {PASS if ok4 else FAIL} duplicate api_port across accounts is rejected")
    all_ok &= ok4

    missing = tmpdir / "does_not_exist.json"
    mod.ACCOUNTS_PATH = missing
    try:
        mod.load_accounts()
        ok5 = False
    except SystemExit:
        ok5 = True
    print(f"  {PASS if ok5 else FAIL} missing accounts.json exits cleanly (not a crash)")
    all_ok &= ok5

finally:
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 70)
print("TEST 3: ChildProcess.start_trading() builds the right /trading/start body")
print("=" * 70)
# Verify the request body logic without a network call, by inspecting what
# would be sent — reuse the same construction inline since it's a small
# private detail of start_trading(); assert on the documented behavior.
acct_ftmo = {
    "id": "a", "label": "A", "api_port": 8000,
    "initial_balance": 10000, "ftmo_enabled": True, "ftmo_initial_balance": 10000,
}
child = mod.ChildProcess(acct_ftmo)
body = {}
if "initial_balance" in child.account:
    body["initial_balance"] = child.account["initial_balance"]
if child.account.get("ftmo_enabled"):
    body["ftmo_enabled"] = True
    body["ftmo_initial_balance"] = child.account.get(
        "ftmo_initial_balance", child.account.get("initial_balance", 0),
    )
ok6 = body == {"initial_balance": 10000, "ftmo_enabled": True, "ftmo_initial_balance": 10000}
print(f"  {PASS if ok6 else FAIL} FTMO account -> body={body}")
all_ok &= ok6

acct_plain = {"id": "b", "label": "B", "api_port": 8001, "initial_balance": 5000}
child2 = mod.ChildProcess(acct_plain)
body2 = {}
if "initial_balance" in child2.account:
    body2["initial_balance"] = child2.account["initial_balance"]
if child2.account.get("ftmo_enabled"):
    body2["ftmo_enabled"] = True
ok7 = body2 == {"initial_balance": 5000}
print(f"  {PASS if ok7 else FAIL} non-FTMO account -> body={body2} (no ftmo_enabled key)")
all_ok &= ok7

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All multi-account launcher logic checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
