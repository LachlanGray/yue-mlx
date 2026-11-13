"""Opt-in actual web + Metal smoke. Isolated history; never touches live queue."""
from pathlib import Path
import threading
import time

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright, expect

from yue2_mlx.web import Studio, ThreadingHTTPServer, handler


def main():
    output = Path('outputs/semantic-ui-check')
    output.mkdir(exist_ok=False)
    studio = Studio(output/'jobs', 'models/yue2-8bit', 'models/yue2-vae')
    server = ThreadingHTTPServer(('127.0.0.1',0),handler(studio))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
            page=browser.new_page(viewport=dict(width=1440,height=1100))
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            page.locator('[data-mode="semantic-audio"]').click()
            page.locator('#semantic-file').set_input_files('outputs/generation-smoke/audio.flac')
            expect(page.locator('#semantic-button')).to_be_enabled(timeout=10000)
            page.locator('#semantic-style').fill('English, piano pop, warm lead vocal')
            for adapter in ['on','off']:
                page.locator('#semantic-adapter').select_option(adapter)
                with page.expect_response('**/api/jobs'):
                    page.locator('#semantic-button').click()
                expect(page.locator('#semantic-button')).to_be_enabled()
            page.screenshot(path=str(output/'queued.png'),full_page=True)
            previous=None;deadline=time.monotonic()+300
            while time.monotonic()<deadline:
                state=studio.snapshot()
                report=[(j['settings']['nar_adapter'],j['status'],j['progress']) for j in state['jobs']]
                if report!=previous:print(report,flush=True);previous=report
                if len(state['jobs'])==2 and all(j['status'] in ('done','failed') for j in state['jobs']):break
                page.wait_for_timeout(1000)
            assert all(j['status']=='done' for j in state['jobs']),state
            expect(page.locator('.job-status').first).to_have_text('Ready',timeout=10000)
            assert page.locator('.job-media audio').count()==6
            page.screenshot(path=str(output/'results.png'),full_page=True)
            assert not errors,errors
            for job in state['jobs']:
                file=studio.root/job['id']/'output/semantic-audio.wav'
                x,sr=sf.read(file)
                assert sr==48000 and x.shape==(383936,2) and np.isfinite(x).all()
                print('Adapter',job['settings']['nar_adapter'],file,'peak GiB',job['peak_gib'],flush=True)
            browser.close()
    finally:
        studio.close();server.shutdown();server.server_close()
    print('Actual Semantic lab UI, queue, adapter on/off and playback elements passed.')


if __name__=='__main__':main()
