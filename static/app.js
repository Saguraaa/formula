const $ = id => document.getElementById(id);
let selected = null, previewUrl = null, result = null, filter = 'all', busy = false;
const names = {correct:'计算正确', incorrect:'计算错误', uncertain:'待确认'};
const reasons = {empty_or_unanswered:'未识别到文字，或算式尚未填写答案', unsupported_characters_or_length:'包含暂不支持的字符或过长的算式', ZeroDivisionError:'出现除以零，需人工确认', SyntaxError:'文字无法解析为完整算式', ValueError:'算式格式不支持或缺少唯一等号', multiple_text_lines:'存在多行文字，需确认阅读顺序'};
reasons.expression_too_long='算式超过 200 个字符，暂不支持自动计算';
reasons.unsupported_characters='OCR 结果含不支持的字符，请核对原图';
function node(tag, cls, text) { const el=document.createElement(tag); if(cls) el.className=cls; if(text !== undefined) el.textContent=text; return el; }
function message(text, error=false) { $('message').textContent=text; $('message').className=error?'error':''; }
function resetResult() {
  result=null; filter='all'; document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('active',b.dataset.filter==='all'); b.setAttribute('aria-pressed',String(b.dataset.filter==='all'));});
  for(const key of ['total','correct','incorrect','uncertain']) $(key).textContent='—';
  $('download').hidden=true; $('original-link').hidden=true; $('timing').textContent='';
  $('results').replaceChildren(node('div','empty','点击「开始识别」，查看这张图片的结果。'));
}
function select(file) {
  if(busy || !file) return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)) {message('请选择 JPG、PNG 或 WEBP 图片。',true);return;}
  if(file.size>20*1024*1024) {message('图片超过 20 MB，请先压缩后再试。',true);return;}
  selected=file;resetResult(); if(previewUrl) URL.revokeObjectURL(previewUrl);previewUrl=URL.createObjectURL(file);
  const img=node('img');img.src=previewUrl;img.alt='待识别的作业原图';$('preview').replaceChildren(img);
  $('file-label').textContent=file.name;$('run').disabled=false;message('图片已就绪，点击「开始识别」。');
}
$('drop').onclick=()=>$('file').click();
$('file').onchange=e=>select(e.target.files[0]);
for(const name of ['dragenter','dragover']) $('drop').addEventListener(name,e=>{e.preventDefault();if(!busy)$('drop').classList.add('dragging');});
for(const name of ['dragleave','drop']) $('drop').addEventListener(name,e=>{e.preventDefault();$('drop').classList.remove('dragging');});
$('drop').addEventListener('drop',e=>select(e.dataTransfer.files[0]));
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
function render() {
  if(!result)return;const container=$('results');container.replaceChildren();
  const rows=result.formulas.filter(row=>filter==='all'||row.status===filter);
  if(!rows.length){container.append(node('div','empty',result.formulas.length?'当前分类没有算式。':'未检测到算式。请换用清晰、完整的作业图片。'));return;}
  for(const row of rows){
    const card=node('article','card');card.id=`formula-${row.id}`;
    const top=node('div','card-top');top.append(node('strong','',`算式 ${String(row.id).padStart(2,'0')}`),node('span',`badge ${row.status}`,names[row.status]));
    const link=node('a','crop-link');link.href=row.crop_url;link.target='_blank';link.rel='noopener';link.title='打开完整裁图';
    const img=node('img');img.src=row.crop_url;img.alt=`第 ${row.id} 道算式的 YOLO 完整裁图`;img.loading='lazy';link.append(img);
    const details=node('div','details'),text=node('div','text-row');text.append(node('span','','OCR 原文'),node('code','',row.raw_text||'（未识别到文字）'));details.append(text);
    if(row.normalized_text && row.normalized_text!==row.raw_text) details.append(node('p','reason',`计算采用：${row.normalized_text}`));
    const explanation=row.status==='uncertain'?(row.reason==='unsupported_characters'?`OCR 结果含不支持的字符：${(row.unsupported_characters||[]).map(c=>`「${c}」`).join('、')}，请核对原图`:(reasons[row.reason]||'无法可靠计算，请人工确认')): `左侧 = ${row.left_value}　${row.is_correct?'＝':'≠'}　右侧 = ${row.right_value}`;
    details.append(node('p','reason',explanation));
    const conf=node('div','confidence');conf.append(node('span','',`检测置信度 ${(row.detection_confidence*100).toFixed(1)}%`),node('span','',`OCR 置信度 ${(row.ocr_confidence*100).toFixed(1)}%`));details.append(conf);
    card.append(top,link,details);container.append(card);
  }
}
document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});render();});
$('run').onclick=async()=>{
  if(!selected||busy)return;busy=true;resetResult();$('run').disabled=true;$('drop').disabled=true;$('run').classList.add('busy');$('run').textContent='正在识别…';$('results').setAttribute('aria-busy','true');
  const start=Date.now();message('正在检测算式并逐项识别；首次运行会加载模型…');
  const timer=setInterval(()=>message(`正在处理，已等待 ${Math.floor((Date.now()-start)/1000)} 秒。首次加载模型可能较慢…`),1000);
  try{
    const data=new FormData();data.append('image',selected);const response=await fetch('/api/predict',{method:'POST',body:data});
    const payload=await response.json();if(!response.ok)throw Error(payload.error||'识别失败');result=payload;
    for(const key of ['total','correct','incorrect','uncertain']) $(key).textContent=result.counts[key];
    const link=node('a');link.href=result.annotated_url;link.target='_blank';link.rel='noopener';link.title='打开完整检测图';
    const img=node('img');img.src=result.annotated_url;img.alt='所有检测框及判题状态，编号与右侧裁图一致';link.append(img);$('preview').replaceChildren(link);
    $('original-link').href=result.original_url;$('original-link').hidden=false;$('download').href=result.download_url;$('download').hidden=false;
    $('timing').textContent=`耗时 ${result.elapsed_seconds} s`;render();message(`识别完成，共检测到 ${result.counts.total} 个算式。点击裁图可查看原尺寸图片。`);
  }catch(error){message(error.message==='Failed to fetch'?'无法连接本地服务，请确认启动窗口仍在运行。':error.message,true);}
  finally{clearInterval(timer);busy=false;$('run').disabled=false;$('drop').disabled=false;$('run').classList.remove('busy');$('run').textContent='重新识别 ↗';$('results').setAttribute('aria-busy','false');}
};
