"""Optional browser smoke test against a running studio; uses installed Chrome.

Run: .venv/bin/python tests/browser_smoke.py
Requires playwright (development-only). Produces a real VAE session locally.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    output = Path("outputs/web-check")
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1120}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:7860")
        page.wait_for_function("document.getElementById('engine-status').textContent.includes('Ready')")
        page.screenshot(path=str(output / "create.png"), full_page=True)
        # Real drag/drop into the lyrics field, not merely programmatic field fill.
        page.locator("#lyrics-drop").evaluate("""el => {
            const data = new DataTransfer();
            data.items.add(new File(['[Verse]\\nA browser test'], 'lyrics.txt', {type:'text/plain'}));
            el.dispatchEvent(new DragEvent('drop', {dataTransfer:data, bubbles:true}));
        }""")
        page.wait_for_function("document.getElementById('lyrics').value.includes('browser test')")
        page.get_by_role("button", name="Audio lab").click()
        page.locator("#audio-file").set_input_files("outputs/vae-smoke/input-resampled.wav")
        page.wait_for_function("!document.getElementById('reconstruct-button').disabled")
        page.get_by_role("button", name="Reconstruct audio").click()
        page.wait_for_function("document.querySelector('.job[data-status=done] audio') !== null", timeout=120000)
        page.screenshot(path=str(output / "audio-lab.png"), full_page=True)
        player = page.locator(".job[data-status=done] audio").first
        player.evaluate("audio => audio.play()")
        page.wait_for_timeout(300)
        assert player.evaluate("audio => audio.currentTime > 0")
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert not errors, errors
        print("Browser checks passed: lyrics drop, audio upload, real reconstruction, playback, mobile layout, no JS errors.")
        browser.close()


if __name__ == "__main__":
    main()
