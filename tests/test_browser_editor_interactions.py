"""Deterministic browser event-race regressions; headless acceptance is separate."""

import shutil
import subprocess

import pytest
from src.gui.web.muscriptor_result_runtime import MUSCRIPTOR_RESULT_JS


def run_javascript(tmp_path, scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for browser runtime contract tests")
    prelude = r"""
global.document = {documentElement:{},readyState:'loading',addEventListener(){},querySelectorAll(){return []}};
global.MutationObserver = function(){this.observe=function(){}};
global.CustomEvent = function(){};
global.cancelAnimationFrame = function(){};
const pending=[];
global.window = {dispatchEvent(){}, AudioContext: function(){
  this.currentTime=10;
  this.resume=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
}};
"""
    # Expose the closure for deterministic tests only; production has no test hook.
    runtime = MUSCRIPTOR_RESULT_JS.replace(
        "projectPlaybackRate: projectPlaybackRate",
        "projectPlaybackRate: projectPlaybackRate, ResultSession: ResultSession",
    )
    harness = tmp_path / "browser-interaction.cjs"
    harness.write_text(
        prelude + runtime + r"""
const assert=require('node:assert/strict');
function session(){
 const s=Object.create(window.musicToMidiMidiEditorRuntime.ResultSession.prototype);
 Object.assign(s,{m:{referenceBpm:120,duration:8,instruments:[],strings:{play:'Play',pause:'Pause',player_failed:'Failed: {error}'}},
   targetBpm:120,position:0,playing:false,starting:false,playRequestId:0,disposed:false,
   buffers:{},sources:[],gains:{},panners:{},play:{},status:{},raf:0,
   applyMix(){},tick(){},layoutPlayhead(){}});
 return s;
}
(async()=>{
""" + scenario + "\n})().catch(e=>{console.error(e);process.exitCode=1;});",
        encoding="utf-8",
    )
    result = subprocess.run(
        [node, str(harness)], text=True, capture_output=True, encoding="utf-8", timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pause_cancels_pending_start_and_immediate_pause_never_goes_negative(tmp_path):
    run_javascript(
        tmp_path,
        r"""
const s=session();
const start=s.start(); s.pause(); pending.shift().resolve(); await start;
assert.equal(s.playing,false); assert.equal(s.starting,false);
const next=s.start(); pending.shift().resolve(); await next;
assert.equal(s.playing,true); s.pause(); assert.equal(s.position,0);
""",
    )


def test_rapid_toggle_seek_and_audio_resume_failure_have_explicit_final_state(tmp_path):
    run_javascript(
        tmp_path,
        r"""
const s=session();
for(let i=0;i<100;i++){s.toggle();s.toggle();}
pending.splice(0).forEach(p=>p.resolve());
await new Promise(resolve=>setImmediate(resolve));
assert.equal(s.playing,false);assert.equal(s.starting,false);
s.toggle();s.seek(3);s.toggle();
pending.splice(0).forEach(p=>p.resolve());
await new Promise(resolve=>setImmediate(resolve));
assert.equal(s.playing,false);assert.equal(s.position,3);
const failure=s.requestPlayback();pending.shift().reject(new Error('audio device denied'));
await failure;assert.match(s.status.textContent,/audio device denied/);
assert.equal(s.playing,false);assert.equal(s.starting,false);
""",
    )


def test_quantization_half_grid_matches_python_and_api_ties_to_even(tmp_path):
    run_javascript(
        tmp_path,
        r"""
const s=session();s.editing=true;s.quantizeScope='all_tracks';s.quantizeGrid='1/32';
s.m.quantizeGrids=['1/32'];s.selectedIndices=new Set();
s.m.notes=[{start:.03125,end:.125,velocity:90}]; // grid .0625, start 0.5 grid, duration 1.5 grids
s.commitEdit=function(){};
s.quantizeSelected();assert.equal(s.m.notes[0].start,0);assert.equal(s.m.notes[0].end,.125);
""",
    )
