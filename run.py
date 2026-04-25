from __future__ import annotations

import threading
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn


APP_URL = "http://127.0.0.1:8000/scene"
HEALTH_URL = "http://127.0.0.1:8000/api/health"


def wait_and_open_browser() -> None:
    for _ in range(40):
        try:
            with urlopen(HEALTH_URL, timeout=1):
                break
        except (TimeoutError, OSError, URLError):
            time.sleep(0.4)
    try:
        webbrowser.open(APP_URL, new=0, autoraise=True)
    except Exception:
        pass


if __name__ == "__main__":
    threading.Thread(target=wait_and_open_browser, daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
