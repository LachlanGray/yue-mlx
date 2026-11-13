"""Optional real Flow lab browser test; run test server on port 7861 first."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("http://127.0.0.1:7861")
        page.locator('[data-mode="flow-edit"]').click()
        # Exercise actual drop events, including replacement after first upload.
        data = list(Path("outputs/vae-smoke/input-resampled.wav").read_bytes())
        for name in ["source.wav", "replacement.wav"]:
            page.locator("#flow-drop").evaluate("""(el, args) => {
                const dt = new DataTransfer();
                dt.items.add(new File([new Uint8Array(args.bytes)], args.name, {type:'audio/wav'}));
                el.dispatchEvent(new DragEvent('drop', {dataTransfer:dt, bubbles:true}));
            }""", {"bytes": data, "name": name})
            expect(page.locator("#flow-name")).to_have_text(name)
            expect(page.locator("#flow-button")).to_be_enabled()
            expect(page.locator("#flow-drop")).to_be_visible()
        page.locator("#flow-steps").fill("2")
        page.locator("#flow-time").fill("0.1")
        with page.expect_response("**/api/jobs") as response:
            page.locator("#flow-button").click()
        job = response.value.json()
        card = page.locator('.job').filter(has=page.get_by_role("heading", name="replacement.wav"))
        expect(card).to_have_attribute("data-status", "done", timeout=120000)
        expect(card.locator("audio")).to_have_count(3)
        report = page.request.get(f"http://127.0.0.1:7861/files/{job['id']}/result.json").json()
        assert report["mode"] == "mix" and report["start_time"] == .1
        assert report["velocity_evaluations"] == 4
        assert report["latent_rms_change"] > 0
        page.screenshot(path="outputs/web-check/flow-lab.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        print(json.dumps({"job": job["id"], "latent_rms_change": report["latent_rms_change"], "browser": "passed"}))
        browser.close()


if __name__ == "__main__":
    main()
