"""Optional browser regression: repeated audio drops, with mocked API responses.

Run: .venv/bin/python tests/browser_audio_drop.py
Requires development-only playwright and installed Chrome. No running server needed.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright, expect


def main():
    static = Path(__file__).resolve().parents[1] / "src/yue2_mlx/static"
    uploads = []

    def respond(route):
        path = route.request.url.removeprefix("http://studio.test")
        if path == "/api/state":
            route.fulfill(json={"jobs": [], "busy": False, "audio_ready": True})
        elif path == "/api/upload":
            uploads.append(route.request.post_data_buffer)
            route.fulfill(json={"id": str(len(uploads)), "seconds": 1, "sample_rate": 48000})
        else:
            route.fulfill(path=static / ("index.html" if path == "/" else path.lstrip("/")))

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        page = browser.new_page()
        page.route("http://studio.test/**", respond)
        page.goto("http://studio.test/")
        page.get_by_role("button", name="Audio lab").click()
        drop = page.locator("#audio-drop")
        previous_src = None
        for name in ["first.wav", "second.wav", "third.wav"]:
            expect(drop).to_be_visible()
            drop.evaluate("""(el, name) => {
                const data = new DataTransfer();
                data.items.add(new File([name], name, {type:'audio/wav'}));
                el.dispatchEvent(new DragEvent('drop', {dataTransfer:data, bubbles:true}));
            }""", name)
            expect(page.locator("#upload-name")).to_have_text(name)
            expect(page.locator("#reconstruct-button")).to_be_enabled()
            expect(drop).to_be_visible()
            expect(drop.locator("strong")).to_have_text("Drop another audio file to replace")
            src = page.locator("#input-player").get_attribute("src")
            assert src and src != previous_src
            previous_src = src
        assert uploads == [b"first.wav", b"second.wav", b"third.wav"]
        browser.close()
        print("Passed: three consecutive audio drops replace the preview and keep the drop zone visible.")


if __name__ == "__main__":
    main()
