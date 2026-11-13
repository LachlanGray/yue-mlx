"""Real browser, mocked API: upload replacement, queue, history reuse, readiness."""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    static = Path(__file__).resolve().parents[1]/'src/yue2_mlx/static'
    jobs, submissions, errors, uploads = [], [], [], []
    ready = [True]
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page = browser.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        def respond(route):
            path=route.request.url.removeprefix('http://studio.test')
            if path=='/api/state':
                route.fulfill(json=dict(jobs=list(reversed(jobs)),busy=True,queued=len(jobs),generation_ready=True,audio_ready=True,semantic_audio_ready=ready[0]))
            elif path=='/api/upload':
                uploads.append(route.request.post_data_buffer)
                route.fulfill(json=dict(id=f'{len(uploads):032x}',seconds=8,sample_rate=48000))
            elif path.startswith('/uploads/'):
                route.fulfill(body=uploads[int(path.split('/')[-1],16)-1],content_type='audio/wav')
            elif path=='/api/jobs':
                data=route.request.post_data_json;submissions.append(data)
                job=dict(id=f'{100+len(jobs):032x}',kind=data['kind'],title=data['name'],status='queued',
                         queue_position=len(jobs)+1,created='2026-09-15T12:00:00Z',progress='Waiting in queue',files=[],settings=data)
                jobs.append(job);route.fulfill(json=job)
            else:
                route.fulfill(path=static/('index.html' if path=='/' else path.lstrip('/')))
        page.route('http://studio.test/**',respond)
        page.goto('http://studio.test/')
        page.locator('[data-mode="semantic-audio"]').click()
        expect(page.locator('#semantic-form')).to_be_visible()
        expect(page.locator('#flow-form')).to_be_hidden()
        expect(page.locator('#semantic-button')).to_be_disabled()
        for i in range(2):
            page.locator('#semantic-drop').evaluate('''(el,i)=>{
              const dt=new DataTransfer();dt.items.add(new File(['audio'+i],'song'+i+'.wav',{type:'audio/wav'}));
              el.dispatchEvent(new DragEvent('drop',{dataTransfer:dt,bubbles:true}));
            }''',i)
            expect(page.locator('#semantic-name')).to_have_text(f'song{i}.wav')
            page.locator('#semantic-style').fill(f'piano {i}')
            page.locator('#semantic-method').select_option('anchored' if i==0 else 'direct')
            page.locator('#semantic-time').fill('0.25')
            if i==0:page.locator('#semantic-noise').fill('0.1')
            page.locator('#semantic-reference-details').evaluate('(el)=>el.open=true')
            page.locator('#semantic-reference-file').set_input_files(dict(name=f'ref{i}.wav',mimeType='audio/wav',buffer=b'reference'))
            expect(page.locator('#semantic-reference-name')).to_have_text(f'ref{i}.wav')
            page.locator('#semantic-reference-placement').select_option('prefix' if i==0 else 'audio')
            page.locator('#semantic-steps').fill(str(12+i))
            page.locator('#semantic-adapter').select_option('on' if i==0 else 'off')
            expect(page.locator('#semantic-button')).to_be_enabled()
            expect(page.locator('#semantic-button')).to_contain_text('Add to queue')
            with page.expect_response('**/api/jobs'):page.locator('#semantic-button').click()
        assert submissions[0]['upload']!=submissions[1]['upload']
        assert submissions[0]['nar_adapter'] and not submissions[1]['nar_adapter']
        assert submissions[0]['noise_amount']==.1 and submissions[1]['noise_amount'] is None
        assert submissions[0]['mode']=='anchored' and submissions[1]['mode']=='direct'
        assert submissions[0]['reference_upload']!=submissions[1]['reference_upload']
        expect(page.locator('.job-kind').first).to_have_text('Semantic reconstruction')
        expect(page.locator('.saved-settings summary').first).to_contain_text('Stock NAR')
        expect(page.locator('.saved-settings summary').last).to_contain_text('NAR v4 adapter')
        page.locator('.saved-settings summary').last.click()
        page.get_by_role('button',name='Use settings').last.click()
        expect(page.locator('#semantic-name')).to_have_text('song0.wav')
        expect(page.locator('#semantic-style')).to_have_value('piano 0')
        expect(page.locator('#semantic-steps')).to_have_value('12')
        expect(page.locator('#semantic-adapter')).to_have_value('on')
        expect(page.locator('#semantic-method')).to_have_value('anchored')
        expect(page.locator('#semantic-noise')).to_have_value('0.1')
        expect(page.locator('#semantic-reference-placement')).to_have_value('prefix')
        expect(page.locator('#semantic-reference-name')).to_have_text('ref0.wav')
        assert len(submissions)==2
        page.locator('[data-mode="generate"]').click()
        expect(page.locator('#semantic-form')).to_be_hidden()
        ready[0]=False;page.evaluate('refresh()')
        page.locator('[data-mode="semantic-audio"]').click()
        expect(page.locator('#semantic-button')).to_be_disabled()
        expect(page.locator('#semantic-setup')).to_be_visible()
        assert not errors,errors
        browser.close()
    print('Semantic frontend passed: drop/replace, queue, adapter selection, history reuse, missing weights.')


if __name__=='__main__':main()
