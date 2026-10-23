"""
Alerts
======
Real alert delivery channels for TradingMonitor.register_alert().

TradingMonitor fires DRAWDOWN_ALERT / DAILY_LOSS_ALERT / EMERGENCY_STOP /
TRADE_OPENED / TRADE_CLOSED internally (see monitoring/monitor.py —
_check_alerts and record_trade()) but had no channel actually wired to
receive them. TelegramAlertSender.send matches the
register_alert(callback: Callable[[str, dict], None]) contract.

Usage
-----
    from monitoring.alerts import TelegramAlertSender
    from config.settings import settings

    if settings.alerts.telegram_enabled:
        monitor.register_alert(
            TelegramAlertSender(
                settings.alerts.telegram_bot_token,
                settings.alerts.telegram_chat_id,
            ).send
        )
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


class TelegramAlertSender:
    """Sends TradingMonitor alerts to a Telegram chat via the Bot API."""

    _API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str, timeout_s: float = 5.0) -> None:
        self._bot_token = bot_token
        self._chat_id   = chat_id
        self._timeout_s = timeout_s

    def send(self, alert_type: str, data: dict) -> None:
        """Callback signature matching TradingMonitor.register_alert()."""
        if not _REQUESTS_OK:
            logger.warning("Telegram alert skipped — 'requests' not installed.")
            return
        text = self._format(alert_type, data)
        try:
            resp = requests.post(
                self._API_URL.format(token=self._bot_token),
                json={"chat_id": self._chat_id, "text": text},
                timeout=self._timeout_s,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Telegram alert failed (status=%s): %s",
                    resp.status_code, resp.text[:200],
                )
        except Exception as exc:
            # Alert delivery must never crash the trading loop.
            logger.warning("Telegram alert send failed: %s", exc)

    @staticmethod
    def _fmt_money(v) -> str:
        return f"{v:,.2f} USD" if isinstance(v, (int, float)) else "n/a"

    @classmethod
    def _format(cls, alert_type: str, data: dict) -> str:
        if alert_type == "TRADE_OPENED":
            arrow = "🟢 BUY" if data.get("direction") == "BUY" else "🔴 SELL"
            conf = data.get("confidence")
            return (
                f"{arrow} {data.get('symbol')} @ {data.get('price')}\n"
                f"Valeur investie: {cls._fmt_money(data.get('invested_value'))}\n"
                f"size={data.get('size')} conviction={conf if conf is not None else 'n/a'} "
                f"ticket={data.get('ticket')}"
            )
        if alert_type == "TRADE_CLOSED":
            emoji = "✅" if data.get("outcome") == "WIN" else "❌" if data.get("outcome") == "LOSS" else "⚪"
            reason = data.get("exit_reason") or "closed"
            return (
                f"{emoji} CLOSED {data.get('direction')} {data.get('symbol')} — "
                f"Bénéfice: {cls._fmt_money(data.get('pnl'))} ({data.get('outcome')})\n"
                f"Raison: {reason}\n"
                f"Valeur investie: {cls._fmt_money(data.get('invested_value'))}\n"
                f"{data.get('entry_price')} -> {data.get('exit_price')} | "
                f"strategy={data.get('strategy')} duration={data.get('duration_s')}s"
            )
        lines = [f"[quant-fund-ai] {alert_type}"]
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
