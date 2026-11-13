"""History pagination bounds live audio elements and supports cross-page reuse."""
from pathlib import Path
from playwright.sync_api import sync_playwright,expect


def main():
    static=Path(__file__).resolve().parents[1]/'src/yue2_mlx/static'
    jobs=[dict(id=f'{i:032x}',kind='generate',title=f'Song {i}',status='done',created='2026-09-15T12:00:00Z',progress='Ready',
               files=['audio.flac'],settings=dict(style=f'style {i}',lyrics='',steps=32,seed=i)) for i in range(20)]
    jobs.insert(0,dict(id='f'*32,kind='generate',title='Active',status='running',created='2026-09-15T12:00:00Z',progress='load_ar',files=[],settings={}))
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        def route(r):
            path=r.request.url.removeprefix('http://studio.test')
            if path=='/api/state':r.fulfill(json=dict(jobs=jobs,busy=True,generation_ready=True,audio_ready=True,semantic_audio_ready=True))
            elif path.startswith('/files/'):r.fulfill(status=404)
            else:r.fulfill(path=static/('index.html' if path in ('/','/history') else path.lstrip('/')))
        page.route('http://studio.test/**',route);page.goto('http://studio.test/')
        expect(page.locator('.job')).to_have_count(4);expect(page.locator('.job-media audio')).to_have_count(3)
        page.get_by_role('link',name='Browse full history').click()
        expect(page.locator('.job')).to_have_count(6);expect(page.locator('.job-media audio')).to_have_count(6)
        expect(page.locator('#history-page-label')).to_have_text('Page 1 / 4')
        page.locator('#history-next').click();expect(page.locator('#history-page-label')).to_have_text('Page 2 / 4')
        expect(page.locator('.job h3').first).to_have_text('Song 6')
        assert page.evaluate('cards.size')==6
        page.locator('.saved-settings summary').first.click()
        page.get_by_role('button',name='Use settings').first.click()
        expect(page.locator('#style')).to_have_value('style 6')
        expect(page.locator('.job')).to_have_count(4)
        assert not errors,errors;browser.close()
    print('Passed: active + three recent, six per history page, discarded players, cross-page reuse.')


if __name__=='__main__':main()
