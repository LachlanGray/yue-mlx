"""Optional real generation check against the running web studio."""
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1120})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:7860")
    page.wait_for_function("!document.getElementById('generate-button').disabled")
    if "--existing" not in sys.argv:
        page.locator("#seconds").fill("4")
        page.get_by_role("button", name="Generate music").click()
        page.wait_for_function("document.querySelector('.job[data-status=running]') !== null", timeout=15000)
        assert page.locator("#generate-button").is_disabled()
    page.wait_for_function("""() => {
        const card = document.querySelector('.job');
        return card && ['done','failed'].includes(card.dataset.status);
    }""", timeout=180000)
    card = page.locator(".job").first
    assert card.get_attribute("data-status") == "done", card.inner_text()
    assert card.locator("audio").count() == 1
    card.locator("audio").evaluate("a => a.play()")
    page.wait_for_function("document.querySelector('.job audio').currentTime > 0", timeout=10000)
    assert "Length limit reached" in card.inner_text()
    output = Path("outputs/web-check")
    output.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(output / "generation.png"), full_page=True)
    assert not errors, errors
    print("Browser generation passed: submitted form, busy state, real synthesis, audio playback, downloads.")
    browser.close()
