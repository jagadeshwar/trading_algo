"""Fyers daily token login — browser OAuth or fully automated headless flow.

Automated flow (no browser, works on Streamlit Cloud):
  Set these env vars / Streamlit secrets:
    FYERS_CLIENT_ID    — your Fyers login ID (e.g. XA12345)
    FYERS_PIN          — your 4-6 digit Fyers login PIN
    FYERS_TOTP_SECRET  — base32 TOTP secret from your 2FA setup
  Then call get_access_token() — it refreshes silently every day.

Browser OAuth flow (local only):
  python auth.py              # opens browser, saves token
  python auth.py --check      # verify today's token is valid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from loguru import logger

try:
    from fyers_apiv3 import fyersModel
    _FYERS_AVAILABLE = True
except ImportError:
    fyersModel = None  # type: ignore[assignment]
    _FYERS_AVAILABLE = False

load_dotenv()

TOKEN_FILE = Path("fyers_token.txt")
CALLBACK_PORT = 5000
_captured: dict = {}  # shared between HTTP handler and main thread


# ── HTTP callback handler ─────────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        auth_code = params.get("auth_code", [None])[0]
        state = params.get("state", [""])[0]

        if auth_code:
            _captured["auth_code"] = auth_code
            _captured["state"] = state
            body = b"<html><body><h2>Login successful.</h2><p>You can close this tab.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"auth_code not found in callback URL")

    def log_message(self, format, *args):
        pass  # suppress HTTP server access logs


def _wait_for_callback(port: int, timeout: int = 120) -> str:
    """Start local HTTP server, wait for Fyers callback, return auth_code."""
    server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.timeout = timeout
    logger.info("Waiting for Fyers callback on http://127.0.0.1:{} ...", port)
    server.handle_request()  # blocks until one request arrives or timeout
    if "auth_code" not in _captured:
        raise TimeoutError(f"No callback received within {timeout}s. Did you complete the login?")
    return _captured["auth_code"]


# ── Token helpers ─────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()


def auto_login() -> str:
    """Headless token refresh using Fyers' login API — no browser required.

    Uses FYERS_CLIENT_ID + FYERS_PIN + FYERS_TOTP_SECRET to complete the
    OAuth flow programmatically. Safe to call from Streamlit Cloud.
    """
    import requests as _req
    try:
        import pyotp
    except ImportError:
        raise RuntimeError("pyotp is required for auto-login: pip install pyotp")

    client_id   = os.environ.get("FYERS_CLIENT_ID",   "").strip()
    pin         = os.environ.get("FYERS_PIN",         "").strip()
    totp_secret = os.environ.get("FYERS_TOTP_SECRET", "").strip()
    app_id      = os.environ.get("FYERS_APP_ID",      "").strip()
    secret      = os.environ.get("FYERS_SECRET",      "").strip()
    redirect_uri = os.environ.get("FYERS_REDIRECT_URI",
                                  f"http://127.0.0.1:{CALLBACK_PORT}")

    if not all([client_id, pin, totp_secret, app_id, secret]):
        raise RuntimeError(
            "Auto-login needs FYERS_CLIENT_ID, FYERS_PIN, FYERS_TOTP_SECRET, "
            "FYERS_APP_ID, FYERS_SECRET in env / Streamlit secrets."
        )

    s = _req.Session()

    # Step 1 — initiate login
    r = s.post("https://api-t2.fyers.in/vagator/v2/send_login_otp",
               json={"fy_id": client_id, "app_id": "2"}, timeout=15)
    r.raise_for_status()
    rk = r.json()["request_key"]

    # Step 2 — verify TOTP
    r = s.post("https://api-t2.fyers.in/vagator/v2/verify_otp",
               json={"request_key": rk, "otp": pyotp.TOTP(totp_secret).now()},
               timeout=15)
    r.raise_for_status()
    rk = r.json()["request_key"]

    # Step 3 — verify PIN
    r = s.post("https://api-t2.fyers.in/vagator/v2/verify_pin",
               json={"request_key": rk, "identity_type": "pin", "identifier": pin},
               timeout=15)
    r.raise_for_status()
    user_token = r.json()["data"]["access_token"]

    # Step 4 — get auth_code (no callback server needed; read Location header)
    session = fyersModel.SessionModel(
        client_id=app_id, secret_key=secret,
        redirect_uri=redirect_uri,
        response_type="code", grant_type="authorization_code",
    )
    auth_url = session.generate_authcode()
    r = s.get(auth_url, headers={"Authorization": f"Bearer {user_token}"},
              allow_redirects=False, timeout=15)
    loc = r.headers.get("Location", "")
    auth_code = parse_qs(urlparse(loc).query).get("auth_code", [None])[0]
    if not auth_code:
        raise RuntimeError(f"auth_code not found in redirect: {loc!r}")

    # Step 5 — exchange for final access token
    session.set_token(auth_code)
    resp = session.generate_token()
    if resp.get("s") != "ok":
        raise RuntimeError(f"Token generation failed: {resp}")

    access_token = resp["access_token"]
    TOKEN_FILE.write_text(json.dumps({"date": _today(), "access_token": access_token}))
    logger.success("Auto-login successful — token saved")
    return access_token


def _can_auto_login() -> bool:
    return all(os.environ.get(k, "").strip()
               for k in ("FYERS_CLIENT_ID", "FYERS_PIN", "FYERS_TOTP_SECRET",
                         "FYERS_APP_ID", "FYERS_SECRET"))


def get_access_token() -> str:
    """Return a valid token — auto-refreshes silently when credentials allow it.

    Priority:
      1. FYERS_ACCESS_TOKEN env var (manual override / Streamlit secret)
      2. Cached fyers_token.txt dated today
      3. Headless auto_login() if FYERS_CLIENT_ID/PIN/TOTP_SECRET are set
      4. Interactive browser OAuth (local only)
    """
    # 1. Manual override
    env_token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
    if env_token:
        logger.info("Using FYERS_ACCESS_TOKEN from environment")
        return env_token

    # 2. Cached token from today
    if TOKEN_FILE.exists():
        try:
            data = json.loads(TOKEN_FILE.read_text())
            if data.get("date") == _today():
                logger.info("Using cached Fyers token (valid for today)")
                return data["access_token"]
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. Headless auto-login (works on Streamlit Cloud)
    if _can_auto_login():
        logger.info("Cached token missing/stale — attempting headless auto-login")
        return auto_login()

    # 4. Browser OAuth fallback (local only)
    return _refresh_token()


def _refresh_token() -> str:
    app_id = os.environ.get("FYERS_APP_ID")
    secret = os.environ.get("FYERS_SECRET")
    redirect_uri = os.environ.get("FYERS_REDIRECT_URI", f"http://127.0.0.1:{CALLBACK_PORT}")

    if not app_id or not secret:
        logger.error("FYERS_APP_ID and FYERS_SECRET must be set (check your .env file)")
        sys.exit(1)

    session = fyersModel.SessionModel(
        client_id=app_id,
        secret_key=secret,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )

    auth_url = session.generate_authcode()

    print("\n" + "="*60)
    print("  Opening Fyers login in your browser...")
    print("  If it doesn't open automatically, visit:")
    print(f"  {auth_url}")
    print("="*60 + "\n")

    webbrowser.open(auth_url)

    try:
        auth_code = _wait_for_callback(CALLBACK_PORT)
    except TimeoutError as e:
        logger.error("{}", e)
        # Fall back to manual entry
        print("\nAuto-capture timed out. Paste the full redirect URL or just the auth_code:")
        raw = input("> ").strip()
        if "auth_code=" in raw:
            auth_code = parse_qs(urlparse(raw).query).get("auth_code", [None])[0]
        else:
            auth_code = raw

    if not auth_code:
        raise RuntimeError("Failed to obtain auth_code")

    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("s") != "ok":
        raise RuntimeError(f"Token generation failed: {response}")

    access_token = response["access_token"]
    TOKEN_FILE.write_text(json.dumps({"date": _today(), "access_token": access_token}))
    logger.success("Access token saved to {}", TOKEN_FILE)
    return access_token


def get_fyers_client(access_token: str | None = None):  # -> fyersModel.FyersModel
    """Return an authenticated FyersModel instance."""
    if not _FYERS_AVAILABLE:
        raise RuntimeError("fyers-apiv3 is not installed. Live trading unavailable in this environment.")
    token = access_token or get_access_token()
    app_id = os.environ.get("FYERS_APP_ID")
    Path("logs").mkdir(exist_ok=True)
    return fyersModel.FyersModel(
        client_id=app_id,
        token=token,
        log_path="logs/",
        is_async=False,
    )


def verify_token(token: str) -> bool:
    """Quick API call to verify the token is still valid."""
    try:
        client = get_fyers_client(access_token=token)
        resp = client.get_profile()
        return resp.get("s") == "ok"
    except Exception:
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fyers auth token manager")
    parser.add_argument("--check", action="store_true", help="Verify today's token is valid")
    args = parser.parse_args()

    if args.check:
        if TOKEN_FILE.exists():
            data = json.loads(TOKEN_FILE.read_text())
            if data.get("date") == _today():
                ok = verify_token(data["access_token"])
                print("Token status:", "VALID ✓" if ok else "INVALID ✗ — run python auth.py")
                sys.exit(0 if ok else 1)
        print("No token for today — run: python auth.py")
        sys.exit(1)
    else:
        token = get_access_token()
        print(f"\nToken ready: {token[:20]}...")
