"""Visual crop editor with automatic updates, movement, and resize handles."""
import base64
import hashlib
import io

from PIL import Image
from streamlit.components.v2 import component

HTML = """
<div class="crop-ui">
 <div class="actions"><button class="reset">Use whole image</button>
 <span class="status" aria-live="polite">Drag around the character</span></div>
 <canvas tabindex="0" aria-label="Crop editor: drag to select, move inside the box, or resize its corners"></canvas>
</div>
"""
CSS = """
.crop-ui {font-family:inherit;color:var(--st-text-color);}
canvas {display:block;max-width:100%;height:auto;cursor:crosshair;touch-action:none;border-radius:8px;outline-offset:3px;}
.actions {display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap;}
button {padding:7px 12px;border-radius:7px;border:1px solid #888;background:transparent;color:inherit;cursor:pointer;white-space:nowrap;font:inherit;font-size:.85em;}
button:hover {border-color:#14b8a6;}
button:disabled {opacity:.4;cursor:default;}
.status {font-size:.82em;opacity:.8;}
"""
JS = """
export default function(component) {
 const {data,parentElement,setStateValue} = component;
 const canvas=parentElement.querySelector('canvas'),ctx=canvas.getContext('2d');
 const reset=parentElement.querySelector('.reset'),status=parentElement.querySelector('.status');
 let selection=data.selection ?? null,operation=null,ready=false,disposed=false;
 let published=JSON.stringify(selection);
 const image=new Image();
 function draw() {
   if (!ready) return;
   reset.disabled=!selection;
   const scale=canvas.width/Math.max(canvas.getBoundingClientRect().width,1);
   ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);
   if (selection) {
     const [l,t,r,b]=selection;
     ctx.fillStyle='rgba(0,0,0,.55)';
     ctx.fillRect(0,0,canvas.width,t*canvas.height);
     ctx.fillRect(0,b*canvas.height,canvas.width,(1-b)*canvas.height);
     ctx.fillRect(0,t*canvas.height,l*canvas.width,(b-t)*canvas.height);
     ctx.fillRect(r*canvas.width,t*canvas.height,(1-r)*canvas.width,(b-t)*canvas.height);
     ctx.strokeStyle='#2dd4bf';ctx.lineWidth=2*scale;
     ctx.strokeRect(l*canvas.width,t*canvas.height,(r-l)*canvas.width,(b-t)*canvas.height);
     for (const [x,y] of [[l,t],[r,t],[l,b],[r,b]]) {
       ctx.fillStyle='#2dd4bf';ctx.fillRect(x*canvas.width-4*scale,y*canvas.height-4*scale,8*scale,8*scale);
     }
   }
   status.textContent=operation?'Release to update the result':selection?'Move inside · Resize corners':'Drag around the character';
 }
 function point(e) {
   const r=canvas.getBoundingClientRect();
   return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];
 }
 function hit(p) {
   if (!selection) return {type:'draw',anchor:p};
   const [l,t,r,b]=selection,rect=canvas.getBoundingClientRect();
   const corners=[[[l,t],[r,b]],[[r,t],[l,b]],[[l,b],[r,t]],[[r,b],[l,t]]];
   const nearby=corners.filter(([q])=>Math.abs(p[0]-q[0])*rect.width<12 && Math.abs(p[1]-q[1])*rect.height<12);
   nearby.sort((a,b)=>Math.hypot(p[0]-a[0][0],p[1]-a[0][1])-Math.hypot(p[0]-b[0][0],p[1]-b[0][1]));
   if (nearby.length) return {type:'resize',anchor:nearby[0][1]};
   if (p[0]>l && p[0]<r && p[1]>t && p[1]<b) return {type:'move',anchor:p};
   return {type:'draw',anchor:p};
 }
 function update(e) {
   if (!operation) return;
   const p=point(e),a=operation.anchor;
   if (operation.type==='move') {
     const s=operation.before;
     const dx=Math.max(-s[0],Math.min(1-s[2],p[0]-a[0]));
     const dy=Math.max(-s[1],Math.min(1-s[3],p[1]-a[1]));
     selection=[s[0]+dx,s[1]+dy,s[2]+dx,s[3]+dy];
   } else selection=[Math.min(a[0],p[0]),Math.min(a[1],p[1]),Math.max(a[0],p[0]),Math.max(a[1],p[1])];
   draw();
 }
 function valid() {
   const r=canvas.getBoundingClientRect();
   return selection && (selection[2]-selection[0])*r.width>=6 && (selection[3]-selection[1])*r.height>=6;
 }
 function publish() {
   const value=JSON.stringify(selection);
   if (value!==published) {published=value;setStateValue('bounds',selection);}
 }
 canvas.onpointerdown=e=>{
   if (!ready || e.button!==0) return;
   e.preventDefault();const start=point(e);canvas.focus({preventScroll:true});operation={...hit(start),before:selection?selection.slice():null};
   canvas.setPointerCapture(e.pointerId);update(e);
 };
 canvas.onpointermove=e=>{
   if (operation) update(e);
   else if (ready) {const type=hit(point(e)).type;canvas.style.cursor=type==='move'?'move':type==='resize'?'nwse-resize':'crosshair';}
 };
 canvas.onpointerup=e=>{
   if (!operation) return;
   update(e);if (!valid()) selection=operation.before;
   operation=null;if(canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
   draw();publish();
 };
 canvas.onpointercancel=()=>{if(operation) selection=operation.before;operation=null;draw();};
 canvas.onkeydown=e=>{
   if (e.key==='Escape') {selection=data.selection??null;operation=null;draw();return;}
   if (!selection || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) return;
   e.preventDefault();const step=e.shiftKey ? 0.05 : 0.005;
   const dx=e.key==='ArrowLeft'?-step:e.key==='ArrowRight'?step:0;
   const dy=e.key==='ArrowUp'?-step:e.key==='ArrowDown'?step:0;
   const x=Math.max(-selection[0],Math.min(1-selection[2],dx)),y=Math.max(-selection[1],Math.min(1-selection[3],dy));
   selection=[selection[0]+x,selection[1]+y,selection[2]+x,selection[3]+y];draw();
 };
 canvas.onkeyup=e=>{if(e.key.startsWith('Arrow')) publish();};
 reset.onclick=()=>{selection=null;operation=null;draw();publish();};
 image.onload=()=>{if(disposed)return;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;ready=true;draw();};
 image.src=data.image;
 return()=>{
   disposed=true;image.onload=null;
   canvas.onpointerdown=canvas.onpointermove=canvas.onpointerup=canvas.onpointercancel=null;
   canvas.onkeydown=canvas.onkeyup=null;reset.onclick=null;
 };
}
"""

def crop_pixels(bounds,width,height):
    if not isinstance(bounds,(list,tuple)) or len(bounds)!=4:
        raise ValueError("Select a crop area")
    left,top,right,bottom=map(float,bounds)
    if not (0<=left<right<=1 and 0<=top<bottom<=1):
        raise ValueError("Select a valid crop area")
    box=(round(left*width),round(top*height),min(width,round(right*width)),min(height,round(bottom*height)))
    if box[2]<=box[0] or box[3]<=box[1]:
        raise ValueError("Select a larger crop area")
    return box


def visual_crop(image):
    import streamlit as st
    # Register against the current Streamlit runtime, including test runtimes.
    _crop_component=component("character_crop_editor",html=HTML,css=CSS,js=JS)
    thumbnail=image.convert("RGB");thumbnail.thumbnail((720,520),Image.Resampling.LANCZOS)
    if max(thumbnail.size) < 256:
        scale=256/max(thumbnail.size)
        thumbnail=thumbnail.resize((round(thumbnail.width*scale),round(thumbnail.height*scale)),Image.Resampling.NEAREST)
    buffer=io.BytesIO();thumbnail.save(buffer,format="PNG");image_bytes=buffer.getvalue()
    key="character_crop_"+hashlib.sha256(image_bytes).hexdigest()[:16]
    selection=st.session_state.get(key,{}).get("bounds")
    result=_crop_component(key=key,data={"image":"data:image/png;base64,"+base64.b64encode(image_bytes).decode("ascii"),
                                      "selection":selection},default={"bounds":None},on_bounds_change=lambda:None)
    return crop_pixels(result.bounds,image.width,image.height) if result.bounds else None
