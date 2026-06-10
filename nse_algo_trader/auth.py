"""Fyers daily token login with auto-capture callback server.

Flow:
  1. Generates auth URL and opens it in the browser automatically.
  2. Starts a local HTTP server on port 5000.
  3. Fyers redirects back to http://127.0.0.1:5000/?auth_code=... after login.
  4. Server captures the auth_code, exchanges it for an access token, saves it.

Usage:
  python auth.py              # full interactive login
  python auth.py --check      # just verify today's token is valid
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


def get_access_token() -> str:
    """Return today's cached token if valid, otherwise trigger a new browser login.

    On Streamlit Cloud (or any environment without a local browser), set the
    FYERS_ACCESS_TOKEN environment variable / Streamlit secret to bypass OAuth.
    """
    # Cloud / CI: token injected via env var or Streamlit secrets
    env_token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
    if env_token:
        logger.info("Using FYERS_ACCESS_TOKEN from environment")
        return env_token

    if TOKEN_FILE.exists():
        try:
            data = json.loads(TOKEN_FILE.read_text())
            if data.get("date") == _today():
                logger.info("Using cached Fyers token (valid for today)")
                return data["access_token"]
        except (json.JSONDecodeError, KeyError):
            pass
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
