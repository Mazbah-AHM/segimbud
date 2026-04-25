from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_URL = os.environ.get("SEGIMBUD_APP_URL", "http://127.0.0.1:8000").rstrip("/")
EXPECTED_APP_VERSION = "segimbud-ui-2026-04-25-v3"


def probe_json(url: str, path: str) -> dict | None:
    try:
        with urlopen(f"{url}{path}", timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return None


def probe_route(url: str, path: str) -> bool:
    try:
        with urlopen(f"{url}{path}", timeout=2) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def app_is_ready(url: str) -> bool:
    health = probe_json(url, "/api/health")
    return (
        probe_route(url, "/scene")
        and health is not None
        and health.get("app_version") == EXPECTED_APP_VERSION
    )


def wait_for_app(url: str, timeout_seconds: float = 18) -> bool:
    started_at = time.time()
    while time.time() - started_at < timeout_seconds:
        if app_is_ready(url):
            return True
        time.sleep(0.7)
    return False


def launch_app(port: str) -> None:
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def ensure_app_url() -> tuple[str | None, str]:
    if app_is_ready(DEFAULT_APP_URL):
        return DEFAULT_APP_URL, "Connected to the app on port 8000."

    launch_app("8000")
    if wait_for_app(DEFAULT_APP_URL):
        return DEFAULT_APP_URL, "Started the app on port 8000."

    if probe_route(DEFAULT_APP_URL, "/scene"):
        return None, "Port 8000 is responding, but it does not look like the current SegImBud app. Restart the FastAPI server on port 8000."

    return None, "The FastAPI app could not be started on port 8000 automatically."


def fetch_health(url: str | None) -> dict | None:
    if not url:
        return None
    return probe_json(url, "/api/health")


def attempt_browser_handoff(target_url: str) -> None:
    if st.session_state.get("_segimbud_browser_handoff_done"):
        return

    st.session_state["_segimbud_browser_handoff_done"] = True
    try:
        webbrowser.open(target_url, new=0, autoraise=True)
    except Exception:
        pass


def main() -> None:
    st.set_page_config(
        page_title="SegImBud Redirect",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
        <style>
            header, footer {visibility: hidden;}
            [data-testid="stSidebar"] {display: none;}
            .stApp {
                background: #0a0c10;
                color: #d19637;
                font-family: 'JetBrains Mono', monospace;
            }
            .redirect-shell {
                min-height: 72vh;
                display: grid;
                place-items: center;
            }
            .redirect-card {
                width: min(38rem, 100%);
                padding: 2rem;
                border: 1px solid rgba(209, 150, 55, 0.45);
                background: rgba(209, 150, 55, 0.06);
                text-align: center;
            }
            .redirect-title {
                margin: 0 0 0.75rem;
                letter-spacing: 0.22em;
                font-size: 1.15rem;
                color: #d19637;
            }
            .redirect-copy {
                margin: 0;
                color: #b9c2ce;
                line-height: 1.6;
            }
            .redirect-link {
                color: #d19637;
            }
        </style>
    """, unsafe_allow_html=True)

    app_url, message = ensure_app_url()
    target_url = f"{app_url}/scene" if app_url else None

    if target_url:
        attempt_browser_handoff(target_url)
        st.markdown(
            f'<meta http-equiv="refresh" content="0; url={target_url}">',
            unsafe_allow_html=True,
        )
        components.html(
            f"""
            <script>
                const target = {json.dumps(target_url)};
                try {{ window.location.replace(target); }} catch (error) {{}}
                try {{ window.parent.location.replace(target); }} catch (error) {{}}
                try {{ window.top.location.replace(target); }} catch (error) {{}}
                try {{ window.open(target, "_top"); }} catch (error) {{}}
            </script>
            """,
            height=0,
            width=0,
        )
        st.markdown(
            f"""
            <div class="redirect-shell">
                <div class="redirect-card">
                    <h1 class="redirect-title">Redirecting to SegImBud</h1>
                    <p class="redirect-copy">{message}</p>
                    <p class="redirect-copy">
                        If the redirect does not happen automatically,
                        <a class="redirect-link" href="{target_url}" target="_self">open the workspace directly</a>.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="redirect-shell">
            <div class="redirect-card">
                <h1 class="redirect-title">SegImBud could not open automatically</h1>
                <p class="redirect-copy">
                    Start the FastAPI app on port 8000, then open <code>http://127.0.0.1:8000/scene</code>.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
