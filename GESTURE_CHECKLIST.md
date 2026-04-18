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

## Regression checks

- [ ] Middle click does not also trigger right click.
- [ ] Reversing scroll direction does not accidentally click.
- [ ] Right click remains responsive after repeated scroll gestures.
