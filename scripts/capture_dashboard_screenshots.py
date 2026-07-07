"""Capture dashboard screenshots for the README and grant brief.

Uses Playwright headless to visit the running Streamlit dashboard at
http://localhost:8504 and snap several targeted screenshots into docs/img/.
"""

from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8504"


def full_page_and_sections():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=120_000)

        # Wait for LOSO section headline to render — this appears last
        page.wait_for_selector('text=CROSS-SUBJECT TRANSFER RESULTS', timeout=120_000)
        # Give plotly a chance to draw
        page.wait_for_timeout(5000)

        # 1) Full page — long screenshot
        page.screenshot(path=str(OUT / "dashboard_full.png"), full_page=True)
        print(f"Saved {OUT}/dashboard_full.png")

        # 2) Above-the-fold — hero shot with title
        page.evaluate("() => window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "hero.png"), clip={"x": 0, "y": 0, "width": 1400, "height": 900})
        print(f"Saved {OUT}/hero.png")

        # 3) Focus on the LOSO results section (section 5) — this is the money shot
        # Find the "CROSS-SUBJECT TRANSFER RESULTS" text and scroll to it
        try:
            loso_header = page.locator("text=CROSS-SUBJECT TRANSFER RESULTS").first
            loso_header.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            # Compute clip
            box = loso_header.bounding_box()
            page.screenshot(
                path=str(OUT / "loso_results.png"),
                clip={"x": 0, "y": max(0, box["y"] - 20), "width": 1400, "height": 820},
            )
            print(f"Saved {OUT}/loso_results.png")
        except Exception as e:
            print(f"loso_results screenshot failed: {e}")

        # 4) Attention heatmap section (3b)
        try:
            attn = page.locator("text=ATTENTION MAP").first
            attn.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            box = attn.bounding_box()
            page.screenshot(
                path=str(OUT / "attention_map.png"),
                clip={"x": 0, "y": max(0, box["y"] - 20), "width": 1400, "height": 820},
            )
            print(f"Saved {OUT}/attention_map.png")
        except Exception as e:
            print(f"attention_map screenshot failed: {e}")

        # 5) Focused vs Tired — the pharmacological comparison
        try:
            pharm = page.locator("text=PHARMACOLOGICAL DISSOCIATION").first
            pharm.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            box = pharm.bounding_box()
            page.screenshot(
                path=str(OUT / "focused_vs_tired.png"),
                clip={"x": 0, "y": max(0, box["y"] - 20), "width": 1400, "height": 820},
            )
            print(f"Saved {OUT}/focused_vs_tired.png")
        except Exception as e:
            print(f"focused_vs_tired screenshot failed: {e}")

        # 6) Logit lens
        try:
            lens = page.locator("text=LOGIT LENS").first
            lens.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            box = lens.bounding_box()
            page.screenshot(
                path=str(OUT / "logit_lens.png"),
                clip={"x": 0, "y": max(0, box["y"] - 20), "width": 1400, "height": 820},
            )
            print(f"Saved {OUT}/logit_lens.png")
        except Exception as e:
            print(f"logit_lens screenshot failed: {e}")

        # 7) Baseline prediction with real trained probs
        try:
            base = page.locator("text=BASELINE PREDICTION").first
            base.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            box = base.bounding_box()
            page.screenshot(
                path=str(OUT / "baseline_prediction.png"),
                clip={"x": 0, "y": max(0, box["y"] - 20), "width": 1400, "height": 620},
            )
            print(f"Saved {OUT}/baseline_prediction.png")
        except Exception as e:
            print(f"baseline_prediction screenshot failed: {e}")

        browser.close()


if __name__ == "__main__":
    full_page_and_sections()
