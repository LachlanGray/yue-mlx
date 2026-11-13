"""Offline browser regression for history settings and form reuse."""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    static = Path(__file__).resolve().parents[1] / "src/yue2_mlx/static"
    settings = {"kind":"generate", "style":"<script>test</script> piano", "lyrics":"", "abc":"C D E",
                "cot":"melody", "seconds":12, "steps":8, "seed":456}
    job = {"id":"a"*32, "kind":"generate", "title":"Test song", "status":"done", "created":"2026-09-14T00:00:00Z",
           "progress":"Ready", "files":[], "settings":settings}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        def respond(route):
            path = route.request.url.removeprefix('http://studio.test')
            if path == '/api/state':
                route.fulfill(json={"jobs":[job], "busy":False, "generation_ready":True})
            elif path.startswith('/uploads/'):
                route.fulfill(body=b'test',content_type='application/octet-stream')
            elif path == '/api/upload':
                route.fulfill(json={"id":"c"*32,"seconds":1,"sample_rate":48000})
            else:
                route.fulfill(path=static / ('index.html' if path == '/' else path.lstrip('/')))
        page.route('http://studio.test/**', respond)
        page.goto('http://studio.test/')
        expect(page.locator('.saved-settings summary')).to_contain_text('seed 456')
        page.locator('.saved-settings summary').click()
        expect(page.locator('.saved-settings pre')).to_contain_text('<script>test</script> piano')
        page.get_by_role('button', name='Use settings').click()
        for field, value in [('style',settings['style']),('lyrics',''),('abc','C D E'),('cot','melody'),('seconds','12'),('steps','8'),('seed','456')]:
            expect(page.locator('#'+field)).to_have_value(value)
        page.wait_for_timeout(1400)
        assert page.locator('.saved-settings').evaluate('(el)=>el.open')
        job.update(id='b'*32,kind='flow-edit',title='Reference test',settings={
            'mode':'anchored','start_time':.2,'steps':4,'seed':123,'anchor_strength':2,
            'style':'piano','lyrics':'','upload':'a'*32,'name':'target.wav',
            'reference_upload':'d'*32,'reference_name':'reference.wav',
            'reference_placement':'audio','reference_strength':.5})
        expect(page.locator('.job').first.locator('.saved-settings summary')).to_contain_text('ref audio',timeout=5000)
        page.locator('.job').first.locator('.saved-settings summary').click()
        page.locator('.job').first.get_by_role('button',name='Use settings').click()
        expect(page.locator('#reference-name')).to_have_text('reference.wav')
        expect(page.locator('#flow-name')).to_have_text('target.wav')
        expect(page.locator('#reference-placement')).to_have_value('audio')
        expect(page.locator('#reference-strength')).to_have_value('0.5')
        expect(page.locator('#flow-button')).to_be_enabled()
        assert not errors, errors
        browser.close()
    print('Passed: readable saved settings, safe text rendering, exact form reuse, expansion survives polling.')


if __name__ == '__main__':
    main()
