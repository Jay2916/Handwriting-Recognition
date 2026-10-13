"""A visual, dependency-free crop selector using Streamlit's component API."""

import base64
import hashlib
import io

from PIL import Image
from streamlit.components.v2 import component

HTML = """
<div class="crop-ui">
  <p>Drag a box around the whole character, then choose <strong>Apply crop</strong>.</p>
  <canvas aria-label="Drag to select the character area"></canvas>
  <div class="actions"><button class="apply" disabled>Apply crop</button>
    <button class="reset">Clear selection</button><span class="status" aria-live="polite"></span></div>
</div>
"""
CSS = """
.crop-ui {font-family:inherit;color:var(--st-text-color);}
p {margin:0 0 10px;}
canvas {display:block;max-width:100%;height:auto;cursor:crosshair;touch-action:none;border-radius:6px;}
.actions {display:flex;align-items:center;gap:12px;margin-top:10px;flex-wrap:wrap;}
button {padding:8px 14px;border-radius:6px;border:1px solid #888;background:transparent;color:inherit;cursor:pointer;}
button.apply {background:#087f8c;color:white;border-color:#087f8c;}
button:disabled {opacity:.45;cursor:default;}
.status {font-size:.9em;}
"""
JS = """
export default function(component) {
  const {data, parentElement, setStateValue} = component;
  const canvas = parentElement.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const apply = parentElement.querySelector('.apply');
  const reset = parentElement.querySelector('.reset');
  const status = parentElement.querySelector('.status');
  let selection = data.selection ?? null;
  let anchor = null, disposed = false;
  const image = new Image();
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    if (selection) {
      const [l,t,r,b] = selection;
      ctx.fillStyle = 'rgba(0,0,0,.55)';
      ctx.fillRect(0,0,canvas.width,t*canvas.height);
      ctx.fillRect(0,b*canvas.height,canvas.width,(1-b)*canvas.height);
      ctx.fillRect(0,t*canvas.height,l*canvas.width,(b-t)*canvas.height);
      ctx.fillRect(r*canvas.width,t*canvas.height,(1-r)*canvas.width,(b-t)*canvas.height);
      ctx.strokeStyle = '#15e4ef';ctx.lineWidth = 2;
      ctx.strokeRect(l*canvas.width,t*canvas.height,(r-l)*canvas.width,(b-t)*canvas.height);
    }
    apply.disabled = !selection || selection[2]-selection[0] < .005 || selection[3]-selection[1] < .005;
    status.textContent = selection ? 'Selection ready' : 'Drag to select';
  }
  function point(e) {
    const rect = canvas.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)),
            Math.max(0,Math.min(1,(e.clientY-rect.top)/rect.height))];
  }
  canvas.onpointerdown = e => {
    if (e.button !== 0) return;
    e.preventDefault();anchor = point(e);canvas.setPointerCapture(e.pointerId);
    selection = [anchor[0],anchor[1],anchor[0],anchor[1]];draw();
  };
  canvas.onpointermove = e => {
    if (!anchor) return;
    const end = point(e);
    selection = [Math.min(anchor[0],end[0]),Math.min(anchor[1],end[1]),
                 Math.max(anchor[0],end[0]),Math.max(anchor[1],end[1])];draw();
  };
  canvas.onpointerup = e => {
    if (anchor) canvas.onpointermove(e);
    anchor = null;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  canvas.onpointercancel = () => {anchor = null;};
  apply.onclick = () => {if (!apply.disabled) setStateValue('bounds',selection);};
  reset.onclick = () => {anchor = null;selection = null;draw();setStateValue('bounds',null);};
  image.onload = () => {
    if (disposed) return;
    canvas.width = image.naturalWidth;canvas.height = image.naturalHeight;draw();
  };
  image.src = data.image;
  return () => {
    disposed = true;image.onload = null;
    canvas.onpointerdown = canvas.onpointermove = canvas.onpointerup = canvas.onpointercancel = null;
    apply.onclick = reset.onclick = null;
  };
}
"""

_crop_component = component("character_crop", html=HTML, css=CSS, js=JS)


def crop_pixels(bounds, width, height):
    """Translate a normalized visual selection back to source-image pixels."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError("Select a crop area")
    left, top, right, bottom = map(float, bounds)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("Select a valid crop area")
    box = (round(left * width), round(top * height), min(width, round(right * width)),
           min(height, round(bottom * height)))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Select a larger crop area")
    return box


def visual_crop(image):
    import streamlit as st

    thumbnail = image.convert("RGB")
    thumbnail.thumbnail((640, 450), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    # Each upload gets its own selection; an old image's crop cannot carry over.
    key = "character_crop_" + hashlib.sha256(image_bytes).hexdigest()[:16]
    selection = st.session_state.get(key, {}).get("bounds")
    result = _crop_component(
        key=key,
        data={"image": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
              "selection": selection},
        default={"bounds": None}, on_bounds_change=lambda: None,
    )
    return crop_pixels(result.bounds, image.width, image.height) if result.bounds else None
