"""Create a local FYERS access token without printing any credential values."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.secrets import load_local_env, update_local_env


def auth_code_from_value(value: str) -> str:
    parsed = urlparse(value)
    return parse_qs(parsed.query).get("auth_code", [value])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Authorize the local FYERS research downloader and save its daily access token to .env.")
    parser.add_argument("--auth-code", help="Optional code or redirect URL received after FYERS login")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args()
    settings = load_local_env(args.env)
    required = ["FYERS_APP_ID", "FYERS_APP_SECRET", "FYERS_REDIRECT_URI"]
    missing = [key for key in required if not settings.get(key)]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} in {args.env}. Copy .env.example to .env and fill them locally.")
    try:
        from fyers_apiv3 import fyersModel
    except ImportError as exc:
        raise RuntimeError("Install FYERS support: python -m pip install -e .[fyers]") from exc
    session = fyersModel.SessionModel(
        client_id=settings["FYERS_APP_ID"], secret_key=settings["FYERS_APP_SECRET"],
        redirect_uri=settings["FYERS_REDIRECT_URI"], response_type="code", grant_type="authorization_code",
    )
    code = args.auth_code
    if not code:
        url = session.generate_authcode()
        print("Opening FYERS login in your browser. Complete FYERS login, then copy the redirect URL or auth_code and run this command again with --auth-code.")
        webbrowser.open(url)
        return
    session.set_token(auth_code_from_value(code))
    response = session.generate_token()
    token = response.get("access_token") if isinstance(response, dict) else None
    if not token:
        message = response.get("message", "token generation failed") if isinstance(response, dict) else "invalid FYERS response"
        raise RuntimeError(f"FYERS authentication failed: {message}")
    update_local_env("FYERS_ACCESS_TOKEN", token, args.env)
    print("FYERS access token saved locally in .env. It was not printed. Run the coverage probe next.")


if __name__ == "__main__":
    main()
