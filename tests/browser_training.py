"""Training management, upload/remove snapshots, and live tokenizer discovery."""
from pathlib import Path
from playwright.sync_api import sync_playwright,expect


def main():
    static=Path(__file__).resolve().parents[1]/'src/yue2_mlx/static'
    songs=[];jobs=[];submissions=[];uploads=[]
    choices=[dict(id='mothersuperior',name='Mothersuperior v4')]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
        page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        def route(r):
            nonlocal songs
            path=r.request.url.removeprefix('http://studio.test')
            if path=='/api/state':r.fulfill(json=dict(jobs=jobs,busy=True,queued=len(jobs),generation_ready=True,audio_ready=True,semantic_audio_ready=True,tokenizers=choices))
            elif path=='/api/training/songs':
                if r.request.method=='POST':songs=r.request.post_data_json['songs']
                r.fulfill(json=songs)
            elif path=='/api/upload':
                uploads.append(r.request.post_data_buffer);r.fulfill(json=dict(id=f'{len(uploads):032x}',seconds=10,sample_rate=48000))
            elif path=='/api/jobs':
                data=r.request.post_data_json;submissions.append(data)
                if data.get('resume_job'):
                    r.fulfill(json=dict(id='c'*32,status='queued'));return
                job=dict(id='a'*32,kind='train',title=data['name'],status='queued',queue_position=1,progress='Waiting in queue',
                         created='2026-09-15T12:00:00Z',settings=dict(data,songs=list(songs)),files=[])
                jobs.append(job);r.fulfill(json=job)
            elif path.startswith('/api/training/'):
                r.fulfill(json=dict(metrics=[dict(step=i,flow_loss=1/i,regularizer_loss=3/i,validation_loss=.8/i) for i in (1,2,3)],resumable=jobs[0]['status']=='cancelled',step=3))
            else:r.fulfill(path=static/('training.html' if path=='/training' else 'index.html' if path=='/' else path.lstrip('/')))
        page.route('http://studio.test/**',route);page.goto('http://studio.test/training')
        expect(page.locator('#train-button')).to_be_disabled()
        page.locator('#songs-drop').evaluate('''el=>{const dt=new DataTransfer();for(let i=0;i<3;i++)dt.items.add(new File(['audio'+i],'song'+i+'.wav',{type:'audio/wav'}));el.dispatchEvent(new DragEvent('drop',{dataTransfer:dt,bubbles:true}));}''')
        expect(page.locator('#songs button')).to_have_count(3)
        expect(page.locator('#songs button').last).to_be_enabled()
        page.locator('#songs button').last.click();expect(page.locator('#songs button')).to_have_count(2)
        page.locator('#name').fill('My sound');page.locator('#steps').fill('5');page.locator('#save_every').fill('1')
        expect(page.locator('#train-button')).to_contain_text('Add training to queue')
        with page.expect_response('**/api/jobs'):page.locator('#train-button').click()
        assert submissions[0]['parameters']['steps']==5 and submissions[0]['parameters']['train_head']
        assert len(jobs[0]['settings']['songs'])==2
        page.locator('#songs button').first.click();expect(page.locator('#songs button')).to_have_count(1)
        assert len(jobs[0]['settings']['songs'])==2
        choices.append(dict(id='a'*32+':step-00000001',name='My sound · step 1'))
        page.evaluate('refresh()');expect(page.locator('#base option')).to_have_count(2)
        expect(page.locator('#runs svg')).to_have_count(3)
        expect(page.locator('#runs [data-series="moving-average"]')).to_have_count(1)
        chart=page.locator('#runs svg').first
        chart.focus();chart.press('End')
        expect(page.locator('#runs .loss-readout').first).to_contain_text('mean (3 updates): 0.611111')
        chart.press('Home')
        expect(page.locator('#runs .loss-readout').first).to_contain_text('Step 1 · loss 1.000000')
        page.evaluate('refresh()')
        expect(page.locator('#runs .loss-readout').first).to_contain_text('Step 1 · loss 1.000000')
        bounds=chart.bounding_box();page.mouse.move(bounds['x']+bounds['width']*.968,bounds['y']+bounds['height']*.5)
        expect(page.locator('#runs .loss-readout').first).to_contain_text('Step 3 · loss 0.333333')
        assert page.evaluate('''()=>{const e=lossChart(Array.from({length:30},(_,i)=>({step:i+1,flow_loss:i+1})),'flow_loss','Test','blue');return e.querySelector('.loss-readout').textContent;}''').endswith('mean (20 updates): 20.500000')
        chart.evaluate('(el)=>el.blur()');page.mouse.move(0,0)
        jobs[0]['status']='cancelled';page.evaluate('refresh()')
        page.get_by_role('button',name='Resume from step 3').click()
        assert submissions[-1]==dict(kind='train',resume_job='a'*32)
        page.goto('http://studio.test/');page.locator('[data-mode="semantic-audio"]').click()
        expect(page.locator('#semantic-tokenizer option')).to_have_count(2)
        page.locator('#semantic-tokenizer').select_option(choices[-1]['id']);page.evaluate('refresh()')
        expect(page.locator('#semantic-tokenizer')).to_have_value(choices[-1]['id'])
        assert not errors,errors;browser.close()
    print('Training UI passed: multi-drop, remove, queued snapshot, parameters, checkpoint choice, three loss plots, resume submission.')


if __name__=='__main__':main()
