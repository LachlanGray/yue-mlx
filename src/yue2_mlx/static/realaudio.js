let semanticUpload=null, semanticUploading=false, semanticPreviewURL=null;
let semanticReference=null, semanticReferenceUploading=false, semanticReferenceURL=null;
Object.assign(settingNames,{noise_amount:'Noise amount (null = follows t)',tokenizer_choice:'Tokenizer checkpoint'});
function semanticControls(){
    const mode=$('semantic-method').value;
    $('semantic-flow-controls').hidden=mode==='fresh';$('semantic-time').disabled=mode==='fresh';
    $('semantic-noise-controls').hidden=mode==='direct';$('semantic-noise').disabled=mode==='direct'||mode==='fresh';
    $('semantic-anchor-controls').hidden=mode!=='anchored';$('semantic-anchor').disabled=mode!=='anchored';
}
$('semantic-method').onchange=semanticControls;semanticControls();
$('semantic-tokenizer').onchange=()=>{
    const choice=$('semantic-tokenizer').value;
    if((state.ar_adapters||[]).some(item=>item.id===choice)){$('semantic-ar').value=choice;$('semantic-adapter').value='on';}
};
async function uploadSemanticReference(file){
    if(semanticReferenceUploading)return;semanticReferenceUploading=true;semanticReference=null;buttons();
    try{
        const r=await api('/api/upload',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});
        semanticReference={id:r.id,name:file.name};if(semanticReferenceURL)URL.revokeObjectURL(semanticReferenceURL);
        semanticReferenceURL=URL.createObjectURL(file);$('semantic-reference-player').src=semanticReferenceURL;
        $('semantic-reference-name').textContent=file.name;$('semantic-reference-preview').hidden=false;
        if($('semantic-reference-placement').value==='off')$('semantic-reference-placement').value='prefix';
        $('semantic-reference-drop').querySelector('strong').textContent='Drop another reference to replace';error();
    }catch(e){$('semantic-reference-preview').hidden=true;error(e.message);}
    finally{semanticReferenceUploading=false;buttons();}
}
function clearSemanticReference(){
    semanticReference=null;if(semanticReferenceURL)URL.revokeObjectURL(semanticReferenceURL);semanticReferenceURL=null;
    $('semantic-reference-player').removeAttribute('src');$('semantic-reference-player').load();
    $('semantic-reference-preview').hidden=true;$('semantic-reference-placement').value='off';
    $('semantic-reference-drop').querySelector('strong').textContent='Drop reference audio here';buttons();
}
bindDrop($('semantic-reference-drop'),$('semantic-reference-file'),uploadSemanticReference);
$('semantic-reference-remove').onclick=clearSemanticReference;
$('semantic-reference-placement').onchange=buttons;
Object.assign(labels,{load_mert:'Loading MERT audio understanding',mert:'Reading the recording',load_token_head:'Loading semantic tokenizer',tokenize_audio:'Musical tokens extracted'});
Object.assign(settingNames,{nar_adapter:'Real-audio NAR adapter',tokenizer_model:'Tokenizer package'});
function semanticMode(mode){
    $('semantic-form').hidden=mode!=='semantic-audio';
    if(mode==='semantic-audio'){
        $('page-title').innerHTML='The music within.<br>A new rendering.';
        $('page-description').textContent='Extract musical tokens from a recording, then hear them rebuilt.';
    }
}
function semanticButtons(){
    const select=$('semantic-tokenizer'),options=state.tokenizers||[{id:'mothersuperior',name:'Mothersuperior v4'}];
    const signature=JSON.stringify(options.map(o=>[o.id,o.name]));
    if(select.dataset.options!==signature){const chosen=select.value;select.replaceChildren(...options.map(o=>new Option(o.name,o.id)));if(options.some(o=>o.id===chosen))select.value=chosen;select.dataset.options=signature;}
    $('semantic-button').firstChild.textContent=state.busy?'Add to queue ':'Rebuild from tokens ';
    $('semantic-button').disabled=submitting||semanticUploading||semanticReferenceUploading||($('semantic-reference-placement').value!=='off'&&!semanticReference)||!semanticUpload||!state.semantic_audio_ready;
    $('semantic-setup').hidden=Boolean(state.semantic_audio_ready);
}
async function uploadSemantic(file){
    if(semanticUploading)return;semanticUploading=true;semanticUpload=null;error();buttons();
    $('semantic-drop').querySelector('strong').textContent='Uploading your audio…';
    try{
        const result=await api('/api/upload',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});
        semanticUpload={id:result.id,name:file.name};
        if(semanticPreviewURL)URL.revokeObjectURL(semanticPreviewURL);
        semanticPreviewURL=URL.createObjectURL(file);$('semantic-player').src=semanticPreviewURL;
        $('semantic-name').textContent=file.name;$('semantic-preview').hidden=false;
    }catch(e){$('semantic-preview').hidden=true;error(e.message);}
    finally{semanticUploading=false;$('semantic-drop').querySelector('strong').textContent=semanticUpload?'Drop another audio file to replace':'Drop your audio here';buttons();}
}
bindDrop($('semantic-drop'),$('semantic-file'),uploadSemantic);
$('semantic-drop').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('semantic-file').click();}});
$('semantic-form').addEventListener('submit',e=>{
    e.preventDefault();if(!semanticUpload)return;
    submit({kind:'semantic-audio',upload:semanticUpload.id,name:semanticUpload.name,
        tokenizer_choice:$('semantic-tokenizer').value,
        acoustic_scale:Number($('semantic-acoustic-scale').value),ar_choice:$('semantic-ar').value,ar_scale:Number($('semantic-ar-scale').value),
        mode:$('semantic-method').value,start_time:$('semantic-method').value==='fresh'?1:Number($('semantic-time').value),
        noise_amount:$('semantic-noise').disabled||$('semantic-noise').value===''?null:Number($('semantic-noise').value),
        anchor_strength:Number($('semantic-anchor').value),
        reference_upload:semanticReference?.id||'',reference_name:semanticReference?.name||'',
        reference_placement:$('semantic-reference-placement').value,reference_strength:Number($('semantic-reference-strength').value),
        style:$('semantic-style').value,lyrics:$('semantic-lyrics').value,
        nar_adapter:$('semantic-adapter').value==='on',steps:Number($('semantic-steps').value),seed:Number($('semantic-seed').value)});
});
async function reuseSemantic(job){
    if(semanticUploading||semanticReferenceUploading){error('Wait for the current upload to finish.');return;}
    const s=job.settings||{};setMode('semantic-audio');$('semantic-form').reset();
    for(const key of ['style','lyrics','steps','seed'])if(Object.hasOwn(s,key))$('semantic-'+key).value=s[key];
    $('semantic-adapter').value=s.nar_adapter===false?'off':'on';
    $('semantic-tokenizer').value=s.tokenizer_choice||'mothersuperior';
    $('semantic-acoustic-scale').value=s.acoustic_scale??1;$('semantic-ar').value=s.ar_choice||'stock';$('semantic-ar-scale').value=s.ar_scale??1;
    if(!$('semantic-tokenizer').value||!$('semantic-ar').value){error('A saved tokenizer or AR checkpoint is unavailable; choose another checkpoint.');return;}
    $('semantic-method').value=['fresh','mix','direct','anchored'].includes(s.mode)?s.mode:'fresh';
    $('semantic-time').value=s.start_time??.15;$('semantic-noise').value=s.noise_amount??'';
    $('semantic-anchor').value=s.anchor_strength??2;semanticControls();clearSemanticReference();
    $('semantic-reference-strength').value=s.reference_strength??1;
    if(s.reference_placement&&s.reference_placement!=='off'){
        $('semantic-reference-details').open=true;$('semantic-reference-placement').value=s.reference_placement;
        const refUrl=s.reference_upload?`/uploads/${s.reference_upload}`:(job.files||[]).includes('reference-resampled.wav')?`/files/${job.id}/reference-resampled.wav`:null;
        try{if(!refUrl)throw new Error('Upload the reference again.');const r=await fetch(refUrl);if(!r.ok)throw new Error('Reference unavailable; upload it again.');await uploadSemanticReference(new File([await r.blob()],s.reference_name||'reference.wav'));$('semantic-reference-placement').value=s.reference_placement;}catch(e){error(e.message);}
    }
    semanticUpload=null;$('semantic-preview').hidden=true;buttons();
    const url=s.upload?`/uploads/${s.upload}`:(job.files||[]).includes('input-resampled.wav')?`/files/${job.id}/input-resampled.wav`:null;
    try{
        if(!url)throw new Error('Drop the original recording to repeat this session.');
        const r=await fetch(url);if(!r.ok)throw new Error('Original recording unavailable; upload it again.');
        await uploadSemantic(new File([await r.blob()],s.name||'recording.wav'));
    }catch(e){error(e.message);}
    $('page-title').scrollIntoView({behavior:'smooth'});
}
buttons();
