"""Browser note feedback must follow musical time without changing editor state."""

from tests.test_browser_editor_interactions import run_javascript


def test_browser_highlights_active_audible_notes_and_clears_expired_notes(tmp_path):
    run_javascript(
        tmp_path,
        r"""
const s=session(), painted=[];
s.selectedIndices=new Set(); s.scroll={scrollLeft:0}; s.pps=100;
s.hoveredIndex=null;
s.feedbackCanvas={width:800,getContext(){return {
 setTransform(){},clearRect(){painted.length=0;},strokeRect(){},
 fillRect(x,y,width,height){painted.push({x,y,width,height});}
}}};
s.audible=(instrument)=>instrument==='piano';
const notes=[{start:0,end:2,pitch:60,instrument:'piano'},
 {start:1,end:3,pitch:64,instrument:'piano'},
 {start:1,end:3,pitch:48,instrument:'bass'}];
s.feedbackNotes=notes.map((note,index)=>({note,index}));
s.position=1.5; s.drawNoteFeedback(); assert.equal(painted.length,2);
s.position=2; s.drawNoteFeedback(); assert.equal(painted.length,1);
assert.equal(painted[0].x,172);
s.position=3; s.drawNoteFeedback(); assert.equal(painted.length,0);
s.hoveredIndex=2; s.drawNoteFeedback(); assert.equal(painted.length,1);
assert.deepEqual(notes,[{start:0,end:2,pitch:60,instrument:'piano'},
 {start:1,end:3,pitch:64,instrument:'piano'},
 {start:1,end:3,pitch:48,instrument:'bass'}]);
""",
    )


def test_browser_hover_reports_full_note_details_and_never_selects_or_edits(tmp_path):
    run_javascript(
        tmp_path,
        r"""
const s=session();s.selectedIndices=new Set();s.selectedIndex=null;
s.scroll={scrollLeft:0};s.pps=100;s.drag=null;s.drawNoteFeedback=()=>{};
s.m.notes=[{start:0,end:2,pitch:60,velocity:91,instrument:'piano'}];
s.m.instruments=[{id:'piano',label:'Piano'}];
s.m.strings.note_hover='{pitch} {midi} {instrument} {velocity} {start} {end} {duration}';
s.canvas={getBoundingClientRect(){return {left:0,top:0};},setAttribute(){},removeAttribute(){}};
s.noteTooltip={id:'tip',style:{},offsetWidth:200,offsetHeight:50};
window.innerWidth=320;window.innerHeight=600;
const before=JSON.stringify(s.m.notes);
s.updateNoteHover({clientX:120,clientY:339});
assert.equal(s.noteTooltip.hidden,false);assert.equal(s.hoveredIndex,0);
assert.equal(s.noteTooltip.textContent,'C4 60 Piano 91 0.00 2.00 2.00');
assert.equal(s.noteTooltip.style.left,'112px');
assert.equal(s.selectedIndex,null);assert.equal(s.selectedIndices.size,0);
assert.equal(JSON.stringify(s.m.notes),before);
s.updateNoteHover(null);assert.equal(s.hoveredIndex,null);assert.equal(s.noteTooltip.hidden,true);
""",
    )
