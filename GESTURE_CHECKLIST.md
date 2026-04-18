# Gesture Checklist

Use this quick checklist after running:

```powershell
python hand_track.py
```

## Core checks

- [ ] Cursor move: move your hand and verify pointer tracks smoothly.
- [ ] Left click: quick index+thumb pinch triggers one left click.
- [ ] Drag: hold index+thumb pinch for drag, release to drop.
- [ ] Right click: hold middle+thumb pinch long enough to right-click.
- [ ] Middle click: three-finger pinch (index+middle+ring) triggers middle click only.
- [ ] Back action: thumb+pinky pinch-hold triggers app/browser back navigation.

## Scrolling checks

- [ ] Scroll (V-pose): show V-pose and tilt/shift to verify vertical and horizontal scroll.

## Zoom and state checks

- [ ] Zoom mode: touch both index fingertips, move up for zoom in and down for zoom out.
- [ ] Pause/resume: make fist to pause, fist again to resume tracking.
- [ ] Enter key (normal mode): with handwriting mode OFF and one hand visible, ring+thumb pinch-hold (~0.22s) sends Enter.

## Handwriting mode checks (optional: set handwriting_enabled = true)

- [ ] Toggle handwriting mode: with two hands visible, ring+thumb hold on left-side hand turns mode on/off.
- [ ] Draw: right-side index+thumb pinch draws in floating handwriting panel.
- [ ] Clear: left-side middle+thumb hold clears panel.
- [ ] Submit: left-side index+thumb hold inserts recognized text (if OCR installed).
- [ ] Backspace: left-side pinky+thumb hold deletes one character.
- [ ] Exit: left-side ring+thumb hold toggles handwriting mode off.

## Regression checks

- [ ] Middle click does not also trigger right click.
- [ ] Reversing scroll direction does not accidentally click.
- [ ] Right click remains responsive after repeated scroll gestures.
