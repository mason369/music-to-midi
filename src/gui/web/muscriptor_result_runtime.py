"""Project-native browser workbench with the public MuScriptor result controls."""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Mapping

from src.gui.web.track_mixer_runtime import track_file_url
from src.models.gm_instruments import get_instrument_name
from src.models.muscriptor_instruments import (
    MUSCRIPTOR_REPRESENTATIVE_PROGRAMS,
    muscriptor_instrument_label,
)

_COLORS = (
    "#4a9eff",
    "#ff8d66",
    "#73a7ff",
    "#c89bff",
    "#ffd166",
    "#ff70a6",
    "#62d2c3",
    "#b3e35d",
)


def _result_instrument_label(instrument: str, language: str) -> str:
    if instrument in MUSCRIPTOR_REPRESENTATIVE_PROGRAMS or instrument == "drums":
        return muscriptor_instrument_label(instrument, language)
    if instrument.startswith("gm:"):
        try:
            return get_instrument_name(int(instrument.split(":", 1)[1]), language)
        except ValueError:
            pass
    return instrument.replace("_", " ")


def build_muscriptor_result_html(
    state: Mapping[str, object],
    translate: Callable[[str], str],
    language: str,
) -> str:
    detected = [str(item) for item in state.get("detected_instruments", [])]
    selected = [str(item) for item in state.get("selected_instruments", [])]
    ordered = list(selected or detected)
    for instrument in detected:
        if instrument not in ordered:
            ordered.append(instrument)
    instrument_wavs = dict(state.get("instrument_wavs", {}))
    instruments = [
        {
            "id": instrument,
            "label": _result_instrument_label(instrument, language),
            "detected": instrument in detected,
            "color": _COLORS[index % len(_COLORS)],
            "url": (
                track_file_url(instrument_wavs[instrument]) if instrument in instrument_wavs else ""
            ),
        }
        for index, instrument in enumerate(ordered)
    ]
    manifest = {
        "notes": list(state.get("notes", [])),
        "duration": float(state.get("duration", 0.0)),
        "backendLabel": str(state.get("backend_label", "")),
        "sourceTrackName": str(state.get("source_track_name", "")),
        "originalUrl": track_file_url(str(state["audio_path"])),
        "instruments": instruments,
        "downloads": {
            "midi": track_file_url(str(state["midi_path"])),
            "transcription": track_file_url(str(state["transcription_wav"])),
            "stereo": track_file_url(str(state["stereo_mix_wav"])),
        },
        "strings": {
            key: translate(f"muscriptor_result.{key}")
            for key in (
                "play",
                "pause",
                "follow",
                "original",
                "stereo",
                "instruments",
                "not_detected",
                "solo",
                "mute",
                "download",
                "download_midi",
                "download_transcription",
                "download_stereo",
                "ready",
                "linked_source",
                "zoom_help",
            )
        },
    }
    encoded = html.escape(json.dumps(manifest, ensure_ascii=False), quote=False)
    return (
        '<div class="msr-root">'
        f'<pre class="msr-manifest" hidden>{encoded}</pre>'
        '<div class="msr-host"></div>'
        "</div>"
    )


MUSCRIPTOR_RESULT_CSS = r"""
.muscriptor-instrument-selector {
  background:#17243d !important; border:1px solid #2c4f7c !important;
  border-radius:8px !important; padding:10px 12px !important;
}
.muscriptor-instrument-selector label > span:first-child {
  color:#4a9eff !important; font-size:13px !important; font-weight:700 !important;
}
.muscriptor-instrument-selector .info {
  color:#9fb3d9 !important; font-size:12px !important; line-height:1.4 !important;
}
.muscriptor-instrument-selector [data-testid="token"] {
  background:#2a3f5f !important; border:1px solid #4a6d96 !important;
  border-radius:5px !important; color:#e0e0e0 !important; font-size:12px !important;
}
.muscriptor-instrument-selector input {
  color:#e0e0e0 !important; font-size:12px !important; min-height:34px !important;
}
.msr-root { margin: 10px 0; color: #e0e0e0; }
.msr-source { color:#8fc6ff; font-weight:600; background:#122039; border:1px solid #2c4f7c; border-radius:6px; padding:8px 10px; margin-bottom:8px; }
.msr-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:10px; padding:12px; border:1px solid #365f8d; border-radius:8px; background:#17243d; }
.msr-btn { background:#2a3f5f; color:#e0e0e0; border:1px solid #3a4a6a; border-radius:5px; padding:6px 11px; cursor:pointer; }
.msr-btn:hover { background:#3a5a7c; border-color:#4a9eff; }
.msr-btn:disabled { opacity:.4; cursor:default; }
.msr-btn.active { color:#8fc6ff; border-color:#4a9eff; background:#203f68; }
.msr-clock { font-family:monospace; color:#c8d3e6; border:1px solid #3a4a6a; border-radius:4px; background:#16213e; padding:5px 8px; }
.msr-mix { margin-left:auto; display:flex; align-items:center; gap:8px; color:#9aa5ad; }
.msr-grid { display:grid; grid-template-columns:minmax(0,4fr) minmax(220px,1fr); gap:12px; margin-top:12px; }
.msr-roll-scroll { overflow:auto; max-height:650px; border:1px solid #365f8d; border-radius:6px; background:#0f1a2d; scrollbar-color:#3d628e #101b2d; scrollbar-width:thin; }
.msr-roll-scroll::-webkit-scrollbar { width:12px; height:12px; }
.msr-roll-scroll::-webkit-scrollbar-track { background:#101b2d; border-radius:6px; }
.msr-roll-scroll::-webkit-scrollbar-thumb { background:#3d628e; border-radius:6px; border:2px solid #101b2d; }
.msr-roll-scroll::-webkit-scrollbar-thumb:hover { background:#4a9eff; }
.msr-roll-world { position:relative; min-height:616px; }
.msr-roll-viewport { position:sticky; left:0; height:616px; overflow:hidden; }
.msr-roll { display:block; cursor:crosshair; }
.msr-playhead { position:absolute; top:0; bottom:0; width:2px; background:#fff; pointer-events:none; will-change:transform; }
.msr-instruments { border:1px solid #365f8d; border-radius:6px; background:#16213e; padding:12px; align-self:start; }
.msr-instruments h3 { margin:0 0 10px; }
.msr-row { display:flex; align-items:center; gap:8px; padding:6px 4px; }
.msr-row.undetected { opacity:.38; text-decoration:line-through; }
.msr-swatch { width:11px; height:11px; }
.msr-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.msr-row.muted .msr-name { opacity:.2; }
.msr-downloads { display:flex; flex-wrap:wrap; gap:8px; padding:12px 0 0; }
.msr-downloads a { text-decoration:none; }
@media (max-width:760px) { .msr-grid { grid-template-columns:1fr; } .msr-mix { margin-left:0; } }
"""


MUSCRIPTOR_RESULT_JS = r"""
(function () {
  "use strict";
  var sessions = [], sharedContext = null, nextSessionId = 1;
  var bufferCache = {}, bufferPromises = {};
  var LEFT = 72, ROW = 7, HEIGHT = 616, MIN_PPS = 46, MAX_PPS = 368, ZOOM_STEP = 1.15;
  function ctx() { if (!sharedContext) sharedContext = new (window.AudioContext || window.webkitAudioContext)(); return sharedContext; }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function el(tag, cls, text) { var n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=text; return n; }
  function button(text, title) { var n=el("button","msr-btn",text); n.type="button"; if(title)n.title=title; return n; }
  function load(url) {
    if (bufferCache[url]) return Promise.resolve(bufferCache[url]);
    if (bufferPromises[url]) return bufferPromises[url];
    bufferPromises[url] = fetch(url).then(function(r){if(!r.ok)throw new Error("HTTP "+r.status+" "+url);return r.arrayBuffer();})
      .then(function(b){return ctx().decodeAudioData(b);}).then(function(b){bufferCache[url]=b;delete bufferPromises[url];return b;})
      .catch(function(e){delete bufferPromises[url];throw e;});
    return bufferPromises[url];
  }
  function ResultSession(root) {
    this.root=root; this.host=root.querySelector(".msr-host"); this.m={}; this.buffers={}; this.sources=[];
    this.gains={}; this.panners={}; this.position=0; this.startedAt=0; this.playing=false; this.muted=new Set(); this.solo=null; this.mix=.75; this.stereo=false; this.follow=true; this.raf=0;
    this.pps=92; this.drawRaf=0; this.disposed=false; this.ownerId="midi-result-"+(nextSessionId++);
    this.onExternalPlayback=this.handleExternalPlayback.bind(this);
  }
  ResultSession.prototype.init=function(){
    try { this.m=JSON.parse(this.root.querySelector(".msr-manifest").textContent); } catch(e){this.host.textContent=String(e);return;}
    this.build(); window.addEventListener("music-to-midi-playback-start",this.onExternalPlayback); var self=this; var jobs=[load(this.m.originalUrl).then(function(b){self.buffers.original=b;})];
    this.m.instruments.forEach(function(i){if(i.detected&&i.url)jobs.push(load(i.url).then(function(b){self.buffers[i.id]=b;}));});
    Promise.all(jobs).then(function(){if(self.disposed)return;self.play.disabled=false;self.status.textContent=self.m.strings.ready;self.drawStatic();self.layoutPlayhead();}).catch(function(e){if(!self.disposed)self.status.textContent=String(e);});
  };
  ResultSession.prototype.build=function(){
    var self=this,s=this.m.strings;
    if(this.m.sourceTrackName){this.host.appendChild(el("div","msr-source",s.linked_source.replace("{track}",this.m.sourceTrackName).replace("{backend}",this.m.backendLabel)));}
    var bar=el("div","msr-toolbar"); this.play=button(s.play);this.play.disabled=true;this.play.onclick=function(){self.toggle();};bar.appendChild(this.play);
    var follow=button(s.follow);follow.classList.add("active");follow.onclick=function(){self.follow=!self.follow;follow.classList.toggle("active",self.follow);};bar.appendChild(follow);
    this.clock=el("span","msr-clock","0.0s");bar.appendChild(this.clock);this.status=el("span","msr-clock","");bar.appendChild(this.status);
    var mix=el("label","msr-mix");mix.appendChild(document.createTextNode(s.original));this.mixInput=el("input");this.mixInput.type="range";this.mixInput.min="0";this.mixInput.max="1";this.mixInput.step=".01";this.mixInput.value=String(this.mix);this.mixInput.oninput=function(){self.mix=parseFloat(this.value);self.applyMix();};mix.appendChild(this.mixInput);mix.appendChild(document.createTextNode("MIDI"));
    var stereo=el("input");stereo.type="checkbox";stereo.onchange=function(){self.stereo=this.checked;self.mixInput.disabled=self.stereo;self.applyMix();};mix.appendChild(stereo);mix.appendChild(document.createTextNode(s.stereo));bar.appendChild(mix);this.host.appendChild(bar);
    var grid=el("div","msr-grid"),scroll=el("div","msr-roll-scroll"),world=el("div","msr-roll-world"),viewport=el("div","msr-roll-viewport");
    this.canvas=el("canvas","msr-roll");this.playhead=el("div","msr-playhead");viewport.appendChild(this.canvas);viewport.appendChild(this.playhead);world.appendChild(viewport);scroll.appendChild(world);this.scroll=scroll;this.world=world;this.viewport=viewport;this.canvas.onclick=function(e){var r=self.canvas.getBoundingClientRect();self.seek((self.scroll.scrollLeft+e.clientX-r.left-LEFT)/self.pps);};
    scroll.addEventListener("scroll",function(){self.scheduleDraw();self.layoutPlayhead();},{passive:true});
    scroll.addEventListener("wheel",function(e){self.onWheel(e);},{passive:false});
    scroll.title=s.zoom_help;grid.appendChild(scroll);
    var aside=el("aside","msr-instruments");aside.appendChild(el("h3","",s.instruments));this.m.instruments.forEach(function(i,index){var row=el("div","msr-row"+(i.detected?"":" undetected"));row.dataset.instrument=i.id;var sw=el("span","msr-swatch");sw.style.background=i.detected?i.color:"#4b5157";row.appendChild(sw);row.appendChild(el("span","msr-name",i.label));if(!i.detected){row.appendChild(el("small","",s.not_detected));}else{var solo=button("S",s.solo),mute=button(s.mute,s.mute);solo.onclick=function(){self.toggleSolo(i.id);};mute.onclick=function(){self.toggleMute(i.id);};row.appendChild(solo);row.appendChild(mute);i.row=row;i.soloButton=solo;i.muteButton=mute;}aside.appendChild(row);});grid.appendChild(aside);this.host.appendChild(grid);
    var dl=el("div","msr-downloads");[["midi","download_midi"],["transcription","download_transcription"],["stereo","download_stereo"]].forEach(function(spec){var a=el("a","msr-btn",s[spec[1]]);a.href=self.m.downloads[spec[0]];a.download="";dl.appendChild(a);});this.host.appendChild(dl);
    this.resizeObserver=new ResizeObserver(function(){self.layout();});this.resizeObserver.observe(scroll);this.layout();
  };
  ResultSession.prototype.stopSources=function(){this.sources.forEach(function(x){try{x.stop();}catch(e){}try{x.disconnect();}catch(e){}});this.sources=[];this.gains={};this.panners={};};
  ResultSession.prototype.start=function(){var c=ctx(),self=this;if(this.position>=this.m.duration)this.position=0;window.dispatchEvent(new CustomEvent("music-to-midi-playback-start",{detail:{owner:this.ownerId}}));return c.resume().then(function(){if(self.disposed)return;self.stopSources();self.startedAt=c.currentTime-self.position;["original"].concat(self.m.instruments.filter(function(i){return i.detected;}).map(function(i){return i.id;})).forEach(function(id){var b=self.buffers[id];if(!b||self.position>=b.duration)return;var src=c.createBufferSource(),gain=c.createGain(),pan=c.createStereoPanner();src.buffer=b;src.connect(gain);gain.connect(pan);pan.connect(c.destination);src.start(0,self.position);self.sources.push(src);self.gains[id]=gain;self.panners[id]=pan;});self.playing=true;self.applyMix();self.play.textContent=self.m.strings.pause;self.tick();});};
  ResultSession.prototype.pause=function(){if(this.playing)this.position=Math.min(this.m.duration,ctx().currentTime-this.startedAt);this.playing=false;this.stopSources();this.play.textContent=this.m.strings.play;cancelAnimationFrame(this.raf);this.layoutPlayhead();};
  ResultSession.prototype.toggle=function(){if(this.playing)this.pause();else this.start();};
  ResultSession.prototype.seek=function(seconds){var was=this.playing;if(was)this.pause();this.position=clamp(seconds,0,this.m.duration);if(was)this.start();else this.layoutPlayhead();};
  ResultSession.prototype.handleExternalPlayback=function(e){if(e.detail&&e.detail.owner!==this.ownerId)this.pause();};
  ResultSession.prototype.audible=function(id){return !this.muted.has(id);};
  ResultSession.prototype.applyMix=function(){var c=ctx(),t=c.currentTime;if(this.gains.original){this.gains.original.gain.setTargetAtTime(this.stereo?1:1-this.mix,t,.01);this.panners.original.pan.setTargetAtTime(this.stereo?-1:0,t,.01);}var self=this;this.m.instruments.forEach(function(i){if(!self.gains[i.id])return;self.gains[i.id].gain.setTargetAtTime(self.audible(i.id)?(self.stereo?1:self.mix):0,t,.01);self.panners[i.id].pan.setTargetAtTime(self.stereo?1:0,t,.01);});};
  ResultSession.prototype.toggleMute=function(id){this.solo=null;if(this.muted.has(id))this.muted.delete(id);else this.muted.add(id);this.syncRows();};
  ResultSession.prototype.toggleSolo=function(id){if(this.solo===id){this.solo=null;this.muted.clear();}else{this.solo=id;this.muted=new Set(this.m.instruments.filter(function(i){return i.detected&&i.id!==id;}).map(function(i){return i.id;}));}this.syncRows();};
  ResultSession.prototype.syncRows=function(){var self=this;this.m.instruments.forEach(function(i){if(!i.detected)return;var muted=self.muted.has(i.id);i.row.classList.toggle("muted",muted);i.soloButton.classList.toggle("active",self.solo===i.id);i.muteButton.classList.toggle("active",muted);i.muteButton.textContent=self.m.strings.mute;});this.applyMix();this.drawStatic();};
  ResultSession.prototype.tick=function(){if(!this.playing)return;this.position=Math.min(this.m.duration,ctx().currentTime-this.startedAt);if(this.position>=this.m.duration){this.position=this.m.duration;this.pause();return;}if(this.follow){var target=LEFT+this.position*this.pps-this.scroll.clientWidth/2;this.scroll.scrollLeft=clamp(target,0,Math.max(0,this.world.clientWidth-this.scroll.clientWidth));}this.layoutPlayhead();var self=this;this.raf=requestAnimationFrame(function(){self.tick();});};
  ResultSession.prototype.layout=function(){var width=Math.max(320,this.scroll.clientWidth||950),dpr=Math.min(2,window.devicePixelRatio||1);this.viewport.style.width=width+"px";this.world.style.width=Math.max(width,LEFT+this.m.duration*this.pps+80)+"px";this.canvas.style.width=width+"px";this.canvas.style.height=HEIGHT+"px";this.canvas.width=Math.round(width*dpr);this.canvas.height=Math.round(HEIGHT*dpr);this.dpr=dpr;this.drawStatic();this.layoutPlayhead();};
  ResultSession.prototype.scheduleDraw=function(){var self=this;if(this.drawRaf)return;this.drawRaf=requestAnimationFrame(function(){self.drawRaf=0;self.drawStatic();});};
  ResultSession.prototype.onWheel=function(e){var modifier=e.ctrlKey||e.altKey;if(modifier){e.preventDefault();var rect=this.scroll.getBoundingClientRect(),anchorX=clamp(e.clientX-rect.left,0,this.scroll.clientWidth),anchorTime=(this.scroll.scrollLeft+anchorX-LEFT)/this.pps,factor=e.deltaY<0?ZOOM_STEP:1/ZOOM_STEP;this.pps=clamp(this.pps*factor,MIN_PPS,MAX_PPS);this.world.style.width=Math.max(this.scroll.clientWidth,LEFT+this.m.duration*this.pps+80)+"px";this.scroll.scrollLeft=Math.max(0,LEFT+anchorTime*this.pps-anchorX);this.drawStatic();this.layoutPlayhead();return;}if(e.shiftKey){e.preventDefault();this.scroll.scrollLeft+=e.deltaY||e.deltaX;}};
  ResultSession.prototype.drawStatic=function(){if(!this.canvas)return;var p=this.canvas.getContext("2d"),d=this.dpr||1,w=this.canvas.width/d,h=HEIGHT,scrollX=this.scroll.scrollLeft,start=Math.max(0,(scrollX-LEFT)/this.pps),end=Math.min(this.m.duration,(scrollX+w-LEFT)/this.pps);p.setTransform(d,0,0,d,0,0);p.fillStyle="#0f1a2d";p.fillRect(0,0,w,h);for(var pitch=21;pitch<=108;pitch++){var y=(108-pitch)*ROW,black=[1,3,6,8,10].indexOf(pitch%12)>=0;p.fillStyle=black?"#13213a":"#172842";p.fillRect(LEFT,y,w-LEFT,ROW);p.strokeStyle="#2b3d5c";p.beginPath();p.moveTo(LEFT,y);p.lineTo(w,y);p.stroke();p.fillStyle=black?"#23282e":"#e4e8eb";p.fillRect(0,y,LEFT,ROW);if(pitch%12===0){p.fillStyle=black?"#ddd":"#222";p.font="7px monospace";p.fillText("C"+(Math.floor(pitch/12)-1),3,y+6);}}var step=this.pps>=180?.5:(this.pps>=80?1:2),first=Math.max(0,Math.floor(start/step)*step);for(var sec=first;sec<=end+step;sec+=step){var x=LEFT+sec*this.pps-scrollX;p.strokeStyle="#36506f";p.beginPath();p.moveTo(x,0);p.lineTo(x,h);p.stroke();p.fillStyle="#7f94b7";p.font="8px monospace";p.fillText(sec.toFixed(step<1?1:0)+"s",x+3,11);}var self=this;this.m.notes.forEach(function(n){if(n.pitch<21||n.pitch>108||n.end<start||n.start>end)return;var x=LEFT+n.start*self.pps-scrollX,y=(108-n.pitch)*ROW+1,width=Math.max(2,(n.end-n.start)*self.pps),inst=self.m.instruments.find(function(i){return i.id===n.instrument;});p.globalAlpha=self.muted.has(n.instrument)?.12:1;p.fillStyle=inst?inst.color:"#4a9eff";p.fillRect(x,y,width,ROW-2);});p.globalAlpha=1;};
  ResultSession.prototype.layoutPlayhead=function(){if(!this.playhead)return;var x=LEFT+this.position*this.pps-this.scroll.scrollLeft;this.playhead.style.transform="translate3d("+x.toFixed(2)+"px,0,0)";this.playhead.style.visibility=(x>=LEFT&&x<=this.scroll.clientWidth)?"visible":"hidden";this.clock.textContent=this.position.toFixed(1)+"s";};
  ResultSession.prototype.dispose=function(){if(this.disposed)return;this.disposed=true;this.pause();window.removeEventListener("music-to-midi-playback-start",this.onExternalPlayback);if(this.resizeObserver)this.resizeObserver.disconnect();cancelAnimationFrame(this.drawRaf);};
  function scan(){for(var i=sessions.length-1;i>=0;i--){if(!sessions[i].root.isConnected){sessions[i].dispose();sessions.splice(i,1);}}document.querySelectorAll(".msr-root:not([data-msr-init])").forEach(function(root){root.setAttribute("data-msr-init","1");var s=new ResultSession(root);sessions.push(s);s.init();});}
  var timer=0;function schedule(){if(timer)return;timer=setTimeout(function(){timer=0;scan();},40);}new MutationObserver(function(changes){for(var i=0;i<changes.length;i++){for(var j=0;j<changes[i].addedNodes.length;j++){var n=changes[i].addedNodes[j];if(n.nodeType===1&&(n.matches(".msr-root")||n.querySelector(".msr-root"))){schedule();return;}}for(var k=0;k<changes[i].removedNodes.length;k++){var r=changes[i].removedNodes[k];if(r.nodeType===1&&(r.matches(".msr-root")||r.querySelector(".msr-root"))){schedule();return;}}}}).observe(document.documentElement,{childList:true,subtree:true});if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",schedule);else schedule();
})();
"""


def muscriptor_result_head() -> str:
    return f"<style>{MUSCRIPTOR_RESULT_CSS}</style><script>{MUSCRIPTOR_RESULT_JS}</script>"
