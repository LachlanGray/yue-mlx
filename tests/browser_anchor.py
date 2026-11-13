"""Optional browser check for anchored mode; mock API, no model/server required."""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    static = Path(__file__).resolve().parents[1] / "src/yue2_mlx/static"
    requests = []

    def respond(route):
        path = route.request.url.removeprefix("http://studio.test")
        if path == "/api/state":
            route.fulfill(json={"jobs": [], "busy": False, "generation_ready": True, "audio_ready": True})
        elif path == "/api/upload":
            route.fulfill(json={"id": "a"*32, "seconds": 121, "sample_rate": 48000})
        elif path == "/api/jobs":
            requests.append(route.request.post_data_json)
            route.fulfill(json={"id": "b"*32})
        else:
            route.fulfill(path=static / ("index.html" if path == "/" else path.lstrip("/")))

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
        page = browser.new_page()
        page.route("http://studio.test/**", respond)
        page.goto("http://studio.test/")
        page.locator('[data-mode="flow-edit"]').click()
        expect(page.locator("#flow-anchor-controls")).to_be_hidden()
        page.locator("#flow-mode").select_option("anchored")
        expect(page.locator("#flow-anchor-controls")).to_be_visible()
        page.locator("#flow-anchor").fill("25")
        page.locator("#flow-steps").fill("96")
        page.locator("#flow-channels").select_option("independent")
        page.locator('#reference-details summary').click()
        for name in ['reference.wav','replacement.wav']:
            page.locator('#reference-drop').evaluate("""(el,name)=>{
                const dt=new DataTransfer();dt.items.add(new File(['test'],name,{type:'audio/wav'}));
                el.dispatchEvent(new DragEvent('drop',{dataTransfer:dt,bubbles:true}));
            }""",name)
            expect(page.locator('#reference-name')).to_have_text(name)
            expect(page.locator('#reference-drop')).to_be_visible()
        page.locator('#reference-placement').select_option('audio')
        page.locator('#reference-strength').fill('6')
        page.locator("#flow-file").set_input_files({"name": "test.wav", "mimeType": "audio/wav", "buffer": b"test"})
        expect(page.locator("#flow-button")).to_be_enabled()
        with page.expect_response("**/api/jobs"):
            page.locator("#flow-button").click()
        assert requests[0]["mode"] == "anchored" and requests[0]["anchor_strength"] == 25
        assert requests[0]['steps'] == 96
        assert requests[0]["independent_channels"] is True
        assert requests[0]['reference_placement']=='audio' and requests[0]['reference_strength']==6
        assert requests[0]['reference_upload']=='a'*32
        page.locator('#reference-placement').select_option('prefix')
        expect(page.locator('#flow-button')).to_be_enabled()
        with page.expect_response("**/api/jobs"):
            page.locator('#flow-button').click()
        assert requests[1]['reference_placement'] == 'prefix'
        page.locator('#reference-remove').click()
        expect(page.locator('#reference-placement')).to_have_value('off')
        expect(page.locator('#reference-preview')).to_be_hidden()
        page.locator("#flow-mode").select_option("direct")
        expect(page.locator("#flow-anchor")).to_be_disabled()
        expect(page.locator("#flow-anchor-controls")).to_be_hidden()
        page.locator('[data-mode="reconstruct"]').click()
        page.locator("#audio-channels").select_option("independent")
        page.locator("#audio-file").set_input_files({"name": "test.wav", "mimeType": "audio/wav", "buffer": b"test"})
        expect(page.locator("#reconstruct-button")).to_be_enabled()
        with page.expect_response("**/api/jobs"):
            page.locator("#reconstruct-button").click()
        assert requests[2]["kind"] == "reconstruct" and requests[2]["independent_channels"] is True
        browser.close()
    print("Passed: reference drop/replace/remove, both placements, independent channels, and values above former caps.")


if __name__ == "__main__":
    main()
