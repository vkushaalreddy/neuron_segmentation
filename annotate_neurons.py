#!/usr/bin/env python3
import argparse
import os
import sys
import csv
from dataclasses import dataclass, asdict
import numpy as np

# Try tifffile first (best for scientific TIFFs), fall back to imageio
try:
    import tifffile as tiff
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False
    import imageio.v2 as imageio

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Circle

matplotlib.rcParams['figure.max_open_warning'] = 0

@dataclass
class ROI:
    id: int
    frame: int
    x: float
    y: float
    radius_px: float

class Annotator:
    def __init__(self, img_path: str, fixed_radius: float = None):
        self.img_path = img_path
        self.fixed_radius = fixed_radius  # If provided, click centers only; else click-drag
        self.out_dir = os.path.dirname(os.path.abspath(img_path))
        self.basename = os.path.splitext(os.path.basename(img_path))[0]

        # Load image
        self.stack = self._load_image(img_path)
        if self.stack.ndim == 2:
            # (H, W) -> add frame dim of size 1
            self.stack = self.stack[np.newaxis, ...]
        elif self.stack.ndim == 3:
            # Could be (T, H, W) or (H, W, C); handle grayscale stacks only
            if self.stack.shape[-1] in (3,4) and self.stack.dtype != bool:
                raise ValueError("RGB images are not supported. Please provide grayscale TIFF or stack.")
        else:
            raise ValueError(f"Unsupported image shape {self.stack.shape}. Provide 2D or (T,H,W) grayscale.")

        self.T, self.H, self.W = self.stack.shape[0], self.stack.shape[-2], self.stack.shape[-1]
        self.frame_idx = 0
        self.rois: list[ROI] = []
        self._next_id = 1

        # Matplotlib UI
        self.fig, self.ax = plt.subplots(figsize=(8, 8 * self.H / max(self.W, 1)))
        self.im_artist = self.ax.imshow(self.stack[self.frame_idx], cmap='gray', origin='upper')
        self.ax.set_title(self._title_text())
        self.ax.set_axis_off()

        self.current_artist = None
        self.selector = EllipseSelector(
            self.ax,
            onselect=self._on_ellipse_select,
            useblit=True,
            interactive=True,
            button=[1],  # left mouse
            minspanx=2, minspany=2,
            ignore_event_outside=True,
            grab_range=5
        )
        # Constrain the selector to circles by syncing width & height on every draw
        self.cid_draw = self.fig.canvas.mpl_connect('motion_notify_event', self._enforce_circle)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.show_labels = True

        # Instruction text
        help_lines = [
            "[Controls]",
            "  Drag LMB: propose a circle; Press ENTER to commit.",
            "  z: undo last   |   c: clear all",
            f"  n / p: next / prev frame (stack of {self.T})" if self.T > 1 else "  (single image)",
            "  s: save CSV + overlay PNG   |   r: toggle labels   |   q or ESC: quit",
        ]
        self.help_text = self.ax.text(
            0.01, 0.01, "\n".join(help_lines),
            transform=self.ax.transAxes, fontsize=9, color='w',
            ha='left', va='bottom',
            bbox=dict(facecolor='black', alpha=0.35, boxstyle='round,pad=0.4')
        )

        plt.tight_layout()
        plt.show()

    def _title_text(self):
        return f"{os.path.basename(self.img_path)}  —  frame {self.frame_idx+1}/{self.T}   (ROIs: {len(self.rois)})"

    def _load_image(self, path):
        if _HAS_TIFFFILE:
            arr = tiff.imread(path)
        else:
            arr = imageio.imread(path)
        arr = np.asarray(arr)
        # If loaded as (H,W) or (T,H,W). Ensure dtype float for display
        if arr.dtype.kind in "ui":
            # normalize for display but keep underlying coords in pixels
            # Don't scale the data here; imshow can handle ints. Just ensure it's not boolean.
            pass
        return arr

    def _on_ellipse_select(self, eclick, erelease):
        # Called when mouse is released; selector will have extents
        if self.fixed_radius is not None:
            return  # In fixed-radius mode we don't use drag to set radius
        try:
            (x0, x1) = sorted([eclick.xdata, erelease.xdata])
            (y0, y1) = sorted([eclick.ydata, erelease.ydata])
        except TypeError:
            return  # selection outside axes
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = abs(x1 - x0) / 2.0
        ry = abs(y1 - y0) / 2.0
        r = (rx + ry) / 2.0  # enforce circle
        self._preview_circle(cx, cy, r)

    def _enforce_circle(self, event):
        # When user is dragging the ellipse, make it circular visually
        if not self.selector.active or self.selector.extents is None:
            return
        try:
            x0, x1, y0, y1 = self.selector.extents
        except Exception:
            return
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        r = (abs(x1 - x0) + abs(y1 - y0)) / 4.0
        # Update preview
        self._preview_circle(cx, cy, r)

    def _preview_circle(self, cx, cy, r):
        # Remove previous preview
        if self.current_artist is not None:
            try:
                self.current_artist.remove()
            except Exception:
                pass
            self.current_artist = None
        circ = Circle((cx, cy), r, edgecolor='lime', facecolor='none', linewidth=1.5, alpha=0.9)
        self.ax.add_patch(circ)
        self.current_artist = circ
        self.fig.canvas.draw_idle()

    def _commit_current(self):
        # Commit the preview circle as a permanent ROI
        if self.current_artist is None:
            return
        cx, cy = self.current_artist.center
        r = self.current_artist.radius
        roi = ROI(id=self._next_id, frame=self.frame_idx, x=float(cx), y=float(cy), radius_px=float(r))
        self.rois.append(roi)
        self._next_id += 1

        # Convert preview to permanent by re-adding a new Circle with annotation
        self.current_artist.remove()
        self.current_artist = None
        self._draw_roi(roi)
        self.ax.set_title(self._title_text())
        self.fig.canvas.draw_idle()

    def _draw_roi(self, roi: ROI):
        circ = Circle((roi.x, roi.y), roi.radius_px, edgecolor='lime', facecolor='none', linewidth=1.5, alpha=0.9)
        self.ax.add_patch(circ)
        if self.show_labels:
            self.ax.text(roi.x, roi.y, str(roi.id), color='yellow', fontsize=9, ha='center', va='center',
                         bbox=dict(facecolor='black', alpha=0.35, boxstyle='round,pad=0.2'))

    def _redraw_frame(self):
        self.im_artist.set_data(self.stack[self.frame_idx])
        # Clear all patches/text except help_text and image
        artists_to_remove = [a for a in self.ax.artists] if hasattr(self.ax, 'artists') else []
        for txt in self.ax.texts[:]:
            if txt is not self.help_text:
                txt.remove()
        for p in self.ax.patches[:]:
            p.remove()
        if self.current_artist is not None:
            try:
                self.current_artist.remove()
            except Exception:
                pass
            self.current_artist = None
        # Draw ROIs for this frame
        for roi in self.rois:
            if roi.frame == self.frame_idx:
                self._draw_roi(roi)
        self.ax.set_title(self._title_text())
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ('enter',):
            self._commit_current()
        elif event.key == 'z':
            # undo last
            if self.rois:
                last = self.rois.pop()
                print(f"Undid ROI {last.id}")
                # Recompute next id as max+1 to keep unique IDs
                if self.rois:
                    self._next_id = max(r.id for r in self.rois) + 1
                else:
                    self._next_id = 1
                self._redraw_frame()
        elif event.key == 'c':
            self.rois.clear()
            self._next_id = 1
            self._redraw_frame()
        elif event.key == 'n':
            if self.frame_idx < self.T - 1:
                self.frame_idx += 1
                self._redraw_frame()
        elif event.key == 'p':
            if self.frame_idx > 0:
                self.frame_idx -= 1
                self._redraw_frame()
        elif event.key == 's':
            self._save_outputs()
        elif event.key == 'r':
            self.show_labels = not self.show_labels
            self._redraw_frame()
        elif event.key in ('escape', 'q'):
            plt.close(self.fig)

    def _save_outputs(self):
        if not self.rois:
            print("No ROIs to save.")
            return
        csv_path = os.path.join(self.out_dir, f"{self.basename}_neurons.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id','frame','x','y','radius_px'])
            writer.writeheader()
            for roi in self.rois:
                writer.writerow(asdict(roi))
        print(f"Saved CSV: {csv_path}")

        # Save overlay PNG for each frame with at least one ROI
        frames_with_rois = sorted(set(r.frame for r in self.rois))
        for fr in frames_with_rois:
            fig, ax = plt.subplots(figsize=(8, 8 * self.H / max(self.W, 1)))
            ax.imshow(self.stack[fr], cmap='gray', origin='upper')
            ax.set_axis_off()
            for roi in self.rois:
                if roi.frame == fr:
                    circ = Circle((roi.x, roi.y), roi.radius_px, edgecolor='lime', facecolor='none', linewidth=1.5)
                    ax.add_patch(circ)
                    ax.text(roi.x, roi.y, str(roi.id), color='yellow', fontsize=9, ha='center', va='center',
                            bbox=dict(facecolor='black', alpha=0.35, boxstyle='round,pad=0.2'))
            out_png = os.path.join(self.out_dir, f"{self.basename}_overlay_f{fr:04d}.png") if self.T > 1 else os.path.join(self.out_dir, f"{self.basename}_overlay.png")
            fig.tight_layout()
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            print(f"Saved overlay: {out_png}")

def parse_args():
    p = argparse.ArgumentParser(description="Circle neurons on a TIFF (2D or stack) and export x,y,radius to CSV.")
    p.add_argument("tif", help="Path to TIFF image (2D or (T,H,W) grayscale).")
    p.add_argument("--fixed-radius", type=float, default=None,
                   help="Optional fixed circle radius in pixels. If set, simply click to place circles of this radius.")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.tif):
        print(f"File not found: {args.tif}", file=sys.stderr)
        sys.exit(1)
    Annotator(args.tif, fixed_radius=args.fixed_radius)

if __name__ == "__main__":
    main()
