
from pathlib import Path

import numpy as np
from IPython.display import Audio
import html


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light dark;font:16px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;--line:color-mix(in srgb,CanvasText 18%,transparent);--panel:color-mix(in srgb,CanvasText 4%,Canvas)}
*{box-sizing:border-box}body{margin:0;background:Canvas;color:CanvasText}main{max-width:1120px;margin:auto;padding:20px;display:grid;gap:16px}h1{font-size:1.4rem;margin:0}h2{font-size:1rem;margin:0 0 8px}.muted{opacity:.68;font-size:.88rem}.panel{border:1px solid var(--line);border-radius:14px;padding:14px;min-width:0;background:var(--panel)}audio{width:100%;margin-top:12px}canvas{display:block;width:100%;height:145px;touch-action:pan-y;cursor:crosshair}#wave{height:115px}.hint{font-size:.8rem;opacity:.65;margin-top:5px}.maskrow{display:grid;grid-template-columns:130px minmax(0,1fr);gap:10px;align-items:center;margin:12px 0}.maskname{font-size:.9rem;overflow-wrap:anywhere}.maskbar{height:25px;border-radius:5px;background:color-mix(in srgb,CanvasText 6%,Canvas);position:relative;overflow:hidden}.active{position:absolute;top:0;bottom:0;background:#25a579;opacity:.85}.segments{width:100%;border-collapse:collapse}.segments td,.segments th{padding:8px 5px;border-bottom:1px solid var(--line);text-align:left;font-size:.9rem}.segments th{opacity:.65;font-weight:500}.segments button{font:inherit;cursor:pointer;border:1px solid var(--line);background:transparent;border-radius:8px;padding:5px 9px}.segment-list{overflow-x:auto}.empty{opacity:.65;font-size:.9rem}.topline{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}.time{font-variant-numeric:tabular-nums}.tracebox{margin-top:16px}.tracebox:first-child{margin-top:0}.tracehead{display:flex;justify-content:space-between;gap:8px;align-items:baseline}.color-key{color:#3986d9}@media(max-width:540px){main{padding:12px}.panel{padding:11px}.maskrow{grid-template-columns:95px minmax(0,1fr)}canvas{height:130px}.segments td,.segments th{font-size:.82rem}}
</style>
</head>
<body>
<main>
  <header class="topline"><h1>__TITLE__</h1><span id="metadata" class="muted"></span></header>
  <section class="panel">
    <div class="topline"><h2>Audio</h2><span id="clock" class="time muted">0.00 s</span></div>
    <canvas id="wave" aria-label="Audio waveform timeline"></canvas>
    <div class="hint">Tap or click the waveform or any trace to seek. Green areas show detected segments.</div>
    <audio id="player" controls preload="metadata" src="__AUDIO_SRC__"></audio>
    __AUDIO_FULL__
  </section>
  <section class="panel"><h2>Feature traces</h2><div id="traces"></div></section>
  <section class="panel"><h2>Activity masks</h2><div id="masks"></div></section>
    <section class="panel">
        <div class="topline">
            <h2>Detected segments</h2>
            <span id="segmentcount" class="muted"></span>
        </div>
        <div class="segment-list" id="segments">__SEGMENTS__</div>
    </section>
</main>
<script id="debug-data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('debug-data').textContent);
const player=document.getElementById('player');
const NS='#3986d9', GREEN='#25a579', ORANGE='#d97706';
document.getElementById('metadata').textContent=`${D.sr.toLocaleString()} Hz · ${(D.duration/60).toFixed(2)} min · mono`;
const fmt=t=>{t=Math.max(0,t);const m=Math.floor(t/60),s=t-m*60;return `${String(m).padStart(2,'0')}:${s.toFixed(2).padStart(5,'0')}`};
function seekAt(canvas,ev){const r=canvas.getBoundingClientRect();const x=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));player.currentTime=x*D.duration;}
function sizeCanvas(c){const r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=Math.max(1,Math.round(r.width*dpr));c.height=Math.max(1,Math.round(r.height*dpr));const ctx=c.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return {ctx,w:r.width,h:r.height};}
function drawWave(){const c=document.getElementById('wave'),{ctx,w,h}=sizeCanvas(c);ctx.clearRect(0,0,w,h);const mid=h/2;ctx.strokeStyle='rgba(128,140,155,.32)';ctx.beginPath();ctx.moveTo(0,mid);ctx.lineTo(w,mid);ctx.stroke();
 for(const s of D.segments){const x=s.start/D.duration*w,xx=s.end/D.duration*w;ctx.fillStyle='rgba(37,165,121,.17)';ctx.fillRect(x,0,xx-x,h)}
 const a=D.waveform;ctx.strokeStyle=NS;ctx.lineWidth=1;ctx.beginPath();for(let i=0;i<a.length;i++){const x=i/(Math.max(1,a.length-1))*w,y=a[i]*(h*.43);ctx.moveTo(x,mid-y);ctx.lineTo(x,mid+y)}ctx.stroke();drawPlayhead(c,ctx,w,h);}
function drawPlayhead(c,ctx,w,h){const x=(player.currentTime/D.duration)*w;ctx.strokeStyle=ORANGE;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
function renderTrace(t,index){const box=document.createElement('div');box.className='tracebox';const hd=document.createElement('div');hd.className='tracehead';const name=document.createElement('strong');name.textContent=t.name;const range=document.createElement('span');range.className='muted';range.textContent=t.unit||'';hd.append(name,range);const c=document.createElement('canvas');c.setAttribute('aria-label',t.name);box.append(hd,c);document.getElementById('traces').append(box);
 function draw(){const {ctx,w,h}=sizeCanvas(c);ctx.clearRect(0,0,w,h);const left=42,right=8,top=8,bottom=22,pw=w-left-right,ph=h-top-bottom;let ymin=Math.min(...t.y),ymax=Math.max(...t.y);if(t.threshold!==null){ymin=Math.min(ymin,t.threshold);ymax=Math.max(ymax,t.threshold)}if(ymin===ymax){ymin-=1;ymax+=1}const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;const x0=t.x[0],x1=Math.max(x0+1e-9,t.x[t.x.length-1]);const X=x=>left+(x-x0)/(x1-x0)*pw,Y=y=>top+(ymax-y)/(ymax-ymin)*ph;
 ctx.font='11px system-ui';ctx.fillStyle='CanvasText';ctx.globalAlpha=.65;for(let i=0;i<=3;i++){const y=top+ph*i/3,val=ymax-(ymax-ymin)*i/3;ctx.strokeStyle='rgba(128,140,155,.25)';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+pw,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(val.toPrecision(4),left-5,y+4)}ctx.globalAlpha=1;
 if(t.threshold!==null){ctx.strokeStyle=ORANGE;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(left,Y(t.threshold));ctx.lineTo(left+pw,Y(t.threshold));ctx.stroke();ctx.setLineDash([])}
 ctx.strokeStyle=NS;ctx.lineWidth=1.25;ctx.beginPath();for(let i=0;i<t.y.length;i++){const x=X(t.x[i]),y=Y(t.y[i]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}ctx.stroke();ctx.fillStyle='CanvasText';ctx.globalAlpha=.65;ctx.textAlign='left';ctx.fillText(fmt(x0),left,h-4);ctx.textAlign='right';ctx.fillText(fmt(x1),left+pw,h-4);ctx.globalAlpha=1;drawPlayhead(c,ctx,w,h)}
 c.addEventListener('click',ev=>seekAt(c,ev));c._redraw=draw;draw();}
function renderMasks(){const root=document.getElementById('masks');if(!D.masks.length){root.innerHTML='<div class="empty">No masks added. Use add_mask(...).</div>';return}for(const m of D.masks){const row=document.createElement('div');row.className='maskrow';const label=document.createElement('div');label.className='maskname';label.textContent=m.name;const bar=document.createElement('div');bar.className='maskbar';for(const [s,e] of m.active){const r=document.createElement('div');r.className='active';r.style.left=`${s/D.duration*100}%`;r.style.width=`${(e-s)/D.duration*100}%`;bar.append(r)}row.append(label,bar);root.append(row);bar.addEventListener('click',ev=>seekAt(bar,ev));}}
function renderSegments(){const root=document.getElementById('segments');document.getElementById('segmentcount').textContent=`${D.segments.length} region(s)`;if(!D.segments.length){root.innerHTML='<div class="empty">No segments added. Use add_segments(...).</div>';return}const table=document.createElement('table');table.className='segments';table.innerHTML='<thead><tr><th>Label</th><th>Start</th><th>End</th><th>Duration</th><th></th></tr></thead>';const body=document.createElement('tbody');D.segments.forEach((s,i)=>{const tr=document.createElement('tr');for(const value of [s.label,fmt(s.start),fmt(s.end),`${(s.end-s.start).toFixed(2)} s`]){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td'),button=document.createElement('button');button.textContent='Play';button.addEventListener('click',()=>{player.currentTime=s.start;const maybe=player.play();if(maybe&&maybe.catch)maybe.catch(()=>{});const stop=()=>{if(player.currentTime>=s.end){player.pause();player.removeEventListener('timeupdate',stop)}};player.addEventListener('timeupdate',stop)});td.append(button);tr.append(td);body.append(tr)});table.append(body);root.append(table)}
player.addEventListener('timeupdate',()=>{document.getElementById('clock').textContent=`${fmt(player.currentTime)} / ${fmt(D.duration)}`;drawWave();document.querySelectorAll('.tracebox canvas').forEach(c=>c._redraw&&c._redraw())});player.addEventListener('seeked',drawWave);window.addEventListener('resize',()=>{drawWave();document.querySelectorAll('.tracebox canvas').forEach(c=>c._redraw&&c._redraw())});
renderMasks();renderSegments();D.traces.forEach(renderTrace);drawWave();
</script>
</body></html>'''


class AudioDebugger:
    def __init__(self, wav: np.ndarray, sr: int, title: str = "Audio Debugger"):
        self.wav = wav
        self.sr = sr
        self._segments = []
        self.title = str(title)

    def save_html(self, path: str | Path = "audio-debug.html",):
        html_doc = _HTML.replace("__TITLE__", html.escape(self.title))
        html_doc = html_doc.replace("__AUDIO_FULL__", Audio(self.wav, rate=self.sr)._repr_html_())
        segments_html = []
        for seg in self._segments:
            chunk = self.wav[
                int(seg["start"] * self.sr):
                int(seg["end"] * self.sr)
            ]
            segments_html.append(Audio(chunk, rate=self.sr)._repr_html_())
        html_doc = html_doc.replace("__SEGMENTS__", "".join(segments_html))
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_doc, encoding="utf-8")
        return output.resolve()

if __name__ == "__main__":
    from iantirta.audio import load_audio
    import librosa
    audio_path = str(Path("D:/IANTIRTACOM/dataset/vocals.wav").expanduser().resolve())
    wav, sr = load_audio(audio_path, num_channels=1)
    out = AudioDebugger(wav, sr)

    hop_length: int = 16
    win_length: int = 64
    sample_size = win_length * sr // 1000
    sample_stride = hop_length * sr // 1000
    intervals = librosa.effects.split(
        wav,
        top_db=40,
        ref=np.max,
        frame_length=sample_size,
        hop_length=sample_stride,
    )

    for start, end in intervals:
        out._segments.append({
            "start": start / sr,
            "end": end / sr,
        })

    out = out.save_html()
    print(out)