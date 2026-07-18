"""Drive the running Loop app with Playwright and capture screenshots.

Assumes Streamlit is already serving at $LOOP_URL (default http://localhost:8531).
Usage: python scripts/screenshots.py
"""
import os
import pathlib
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("LOOP_URL", "http://localhost:8531")
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
OUT = pathlib.Path(os.environ.get("SHOT_DIR", "docs/screenshots"))
OUT.mkdir(parents=True, exist_ok=True)


def settle(page, ms=1400):
    page.wait_for_timeout(ms)


def click_text(page, text):
    page.get_by_text(text, exact=False).first.click()
    settle(page)


def run():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 900, "height": 1200})
        page.goto(URL, wait_until="networkidle")
        settle(page, 2500)

        # Screen 1
        page.screenshot(path=str(OUT / "01_pick.png"), full_page=True)

        # --- Binge eating flow ---
        click_text(page, "Binge eating")
        page.screenshot(path=str(OUT / "02_customise.png"), full_page=True)
        click_text(page, "Show me the curve")
        settle(page, 1800)
        page.screenshot(path=str(OUT / "03_binge_narrated.png"), full_page=True)
        click_text(page, "What if?")
        settle(page, 1800)
        page.screenshot(path=str(OUT / "04_binge_whatif.png"), full_page=True)

        # Restart -> Porn flow
        click_text(page, "Map a different loop")
        settle(page, 1500)
        click_text(page, "Porn")
        click_text(page, "Show me the curve")
        settle(page, 1800)
        page.screenshot(path=str(OUT / "05_porn_narrated.png"), full_page=True)

        browser.close()
        print("screenshots written to", OUT)


if __name__ == "__main__":
    run()
