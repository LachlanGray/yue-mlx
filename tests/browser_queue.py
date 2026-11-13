"""Frontend queue regression, mocked API and real headless Chrome."""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    static = Path(__file__).resolve().parents[1] / 'src/yue2_mlx/static'
    jobs, submissions = [], []
    uploads = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless=True)
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))

        def respond(route):
            path = route.request.url.removeprefix('http://studio.test')
            if path == '/api/state':
                route.fulfill(json=dict(jobs=list(reversed(jobs)), busy=True,
                                        queued=sum(j['status'] == 'queued' for j in jobs),
                                        generation_ready=True, audio_ready=True,tokenizers=[dict(id='custom',name='My trained acoustic model')]))
            elif path == '/api/upload':
                uploads.append(route.request.post_data_buffer)
                route.fulfill(json=dict(id=f'{len(uploads):032x}', seconds=30, sample_rate=48000))
            elif path.startswith('/uploads/'):
                route.fulfill(body=b'audio',content_type='application/octet-stream')
            elif path == '/api/jobs':
                data = route.request.post_data_json
                submissions.append(data)
                job = dict(id=f'{len(jobs)+100:032x}', kind=data['kind'], title=data.get('name', data.get('style')),
                           status='queued', queue_position=len(jobs)+1, created='2026-09-14T12:00:00Z',
                           progress='Waiting in queue', files=[], settings=data)
                jobs.append(job)
                route.fulfill(json=job)
            elif path == '/api/cancel':
                identifier = route.request.post_data_json['id']
                next(j for j in jobs if j['id'] == identifier).update(status='cancelled', progress='Removed from queue.')
                route.fulfill(json=dict(ok=True))
            else:
                route.fulfill(path=static / ('index.html' if path == '/' else path.lstrip('/')))

        page.route('http://studio.test/**', respond)
        page.goto('http://studio.test/')
        expect(page.locator('#generate-button')).to_be_enabled()
        expect(page.locator('#generate-button')).to_contain_text('Add to queue')
        page.locator('[data-mode="flow-edit"]').click()
        page.locator('#flow-acoustic').select_option('custom')
        page.evaluate('refresh()')
        expect(page.locator('#flow-acoustic')).to_have_value('custom')
        for index in range(2):
            page.locator('#flow-drop').evaluate('''(el, i) => {
                const dt = new DataTransfer();
                dt.items.add(new File(['audio'+i], 'song'+i+'.wav', {type:'audio/wav'}));
                el.dispatchEvent(new DragEvent('drop', {dataTransfer:dt, bubbles:true}));
            }''', index)
            expect(page.locator('#flow-name')).to_have_text(f'song{index}.wav')
            page.locator('#flow-time').fill(str(.1 + index * .1))
            expect(page.locator('#flow-button')).to_be_enabled()
            with page.expect_response('**/api/jobs'):
                page.locator('#flow-button').click()
            assert submissions[-1]['acoustic_choice']=='custom'
        expect(page.locator('.job-status').first).to_have_text('Queued · #2')
        assert submissions[0]['upload'] != submissions[1]['upload']
        assert submissions[0]['start_time'] == .1 and submissions[1]['start_time'] == .2
        with page.expect_response('**/api/cancel'):
            page.get_by_role('button', name='Remove from queue').first.click()
        expect(page.locator('.job[data-status="cancelled"] .job-status')).to_have_text('Stopped')
        expect(page.locator('#flow-button')).to_be_enabled()
        page.locator('.job').first.locator('details').evaluate('(el)=>el.open=true')
        page.locator('.job').first.get_by_role('button',name='Use settings').click()
        expect(page.locator('#flow-acoustic')).to_have_value('custom')
        page.locator('[data-mode="generate"]').click()
        page.locator('#generation-acoustic').select_option('custom')
        with page.expect_response('**/api/jobs'):page.locator('#generate-button').click()
        assert submissions[-1]['kind']=='generate' and submissions[-1]['acoustic_choice']=='custom'
        assert not errors, errors
        browser.close()
    print('Passed: submit while busy, replace uploads/settings, queue positions, remove queued job.')


if __name__ == '__main__':
    main()
