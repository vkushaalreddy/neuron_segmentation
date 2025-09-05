# temp1.py — ImageJ-style polygon/freehand annotator with reliable edges, numbering, undo
# Usage:
#   python temp1.py --input img1frame.tif --axes auto --out outputs\roi

import argparse, os, zipfile, numpy as np, pandas as pd
from tifffile import TiffFile, imread, imwrite
import napari
from roifile import ImagejRoi
from skimage.draw import polygon as sk_polygon

# Try dask/zarr for streaming; fall back to numpy if not available
try:
    import dask.array as da, zarr
    USE_DASK = True
except Exception:
    da = zarr = None
    USE_DASK = False


# -------------------- helpers: reductions --------------------
def _reduce(arr, axis, how="max"):
    if USE_DASK and isinstance(arr, da.Array):
        if how == "max":    return da.max(arr, axis=axis)
        if how == "mean":   return da.mean(arr, axis=axis)
        if how == "median": return da.median(arr, axis=axis)
    else:
        if how == "max":    return np.max(arr, axis=axis)
        if how == "mean":   return np.mean(arr, axis=axis)
        if how == "median": return np.median(arr, axis=axis)
    raise ValueError("unknown reduce method")

def _expand(arr, axis=0):
    return da.expand_dims(arr, axis) if (USE_DASK and isinstance(arr, da.Array)) else np.expand_dims(arr, axis)


# -------------------- core loader --------------------
def _detect_series(path):
    with TiffFile(path) as tf:
        s = tf.series[0]
        axes = getattr(s, "axes", None)
        store = None
        if USE_DASK:
            try: store = s.aszarr()
            except Exception: store = None
        shape = s.shape
    return shape, axes, store

def open_stack_any(path, axes="auto", zproject="max", c_index=None, c_reduce="max"):
    _, detected_axes, store = _detect_series(path)

    if USE_DASK and store is not None:
        arr = da.from_zarr(zarr.open(store, mode="r"))
    else:
        np_arr = imread(path)
        arr = da.from_array(np_arr, chunks="auto") if USE_DASK else np_arr

    axes = (detected_axes or "").upper() if (axes.lower() == "auto") else axes.upper()

    if not axes:
        if arr.ndim == 2: axes = "YX"
        elif arr.ndim == 3: axes = "TYX"
        elif arr.ndim == 4: axes = "TZYX"
        else: raise ValueError(f"Could not infer axes for shape {arr.shape}. Pass --axes explicitly.")

    keep = list(axes)
    unknown = [a for a in keep if a not in list("TZCYX")]
    for a in unknown:
        idx = keep.index(a)
        arr = _reduce(arr, axis=idx, how="max")
        keep.pop(idx)
    axes = "".join(keep)

    if "C" in axes:
        cpos = axes.index("C")
        if c_index is not None:
            slicer = [slice(None)] * arr.ndim; slicer[cpos] = c_index
            arr = arr[tuple(slicer)]
        else:
            arr = _reduce(arr, axis=cpos, how=c_reduce)
        axes = axes.replace("C", "")

    if "Z" in axes:
        zpos = axes.index("Z")
        arr = _reduce(arr, axis=zpos, how=zproject)
        axes = axes.replace("Z", "")

    if "T" not in axes:
        arr = _expand(arr, axis=0)
        axes = "T" + axes

    order = [axes.index("T"), axes.index("Y"), axes.index("X")]
    arr = arr.transpose(order)

    if USE_DASK and isinstance(arr, da.Array):
        arr = arr.rechunk({0: 1, 1: 512, 2: 512})

    return arr  # (T, Y, X)


# -------------------- ROI export (with styled edges + numeric names) --------------------
def save_roizip_for_frame(zip_path, shapes_layer, frame_index,
                          stroke_color=(255, 255, 0), stroke_width=1.4):
    """Save styled, numbered ROIs for a frame."""
    shape_data  = list(shapes_layer.data)
    shape_types = list(shapes_layer.shape_type)
    feats = shapes_layer.features
    frame_tags  = list(feats["t"]) if "t" in feats else [0]*len(shape_data)
    labels_col  = list(feats["label"]) if "label" in feats else ["" for _ in shape_data]

    k = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, (pts, stype, t) in enumerate(zip(shape_data, shape_types, frame_tags)):
            if int(t) != int(frame_index): continue
            pts = np.asarray(pts, np.float32)
            if pts.shape[0] < 3: continue  # ignore unfinished clicks
            if stype == "path" and (pts[0] != pts[-1]).any():
                pts = np.vstack([pts, pts[0]])
            xy = np.column_stack([pts[:, 1], pts[:, 0]]).astype(np.float32)

            roi = ImagejRoi.frompoints(xy)
            # --- AFTER ---
            try:
                import struct
                roi.stroke_width = int(round(float(stroke_width)))
                
                # Convert color tuples to the required bytes format
                sc = stroke_color
                roi.stroke_color = struct.pack('4B', int(sc[0]), int(sc[1]), int(sc[2]), 255)
                roi.fill_color = struct.pack('4B', 0, 0, 0, 0) # Use a transparent fill color
                
            except Exception as e:
                print(f"Warning: Could not set ROI style properties. Error: {e}")
                pass

            number = str(labels_col[i]) if str(labels_col[i]) else str(k + 1)
            roi.name = number

            zf.writestr(f"{roi.name}.roi", roi.tobytes())
            k += 1


def mask_for_frame(H, W, shapes_layer, frame_index):
    # Use uint16 to allow for more than 255 neurons per frame
    mask = np.zeros((H, W), np.uint16) 
    
    feats = shapes_layer.features
    shape_data = list(shapes_layer.data)
    
    if "t" not in feats or len(feats) == 0 or not feats["t"].notna().any():
        return mask # Return empty mask if no features

    frame_indices = np.where(feats["t"].astype(int) == int(frame_index))[0]

    for i in frame_indices:
        # --- ADD THIS SAFETY CHECK ---
        # Ensure the index 'i' is valid for BOTH lists before proceeding
        if i >= len(shape_data) or i >= len(feats):
            continue
        # --- END OF CHECK ---

        pts = np.asarray(shape_data[i], np.float32)
        if pts.shape[0] < 3: continue

        label_str = feats.at[i, "label"]
        
        if label_str and label_str.isdigit():
            instance_id = int(label_str)
            rr, cc = sk_polygon(pts[:, 0], pts[:, 1], shape=mask.shape)
            mask[rr, cc] = instance_id
            
    return mask


# -------------------- GUI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .tif/.ome.tif")
    ap.add_argument("--axes", default="auto",
                    help="auto (default) or a string like YX, TYX, TZYX, TCZYX, etc.")
    ap.add_argument("--zproject", default="max", choices=["max", "median", "mean"],
                    help="How to collapse Z (if present).")
    ap.add_argument("--c-index", type=int, default=None,
                    help="Select a specific channel index (0-based). If omitted, channels are reduced.")
    ap.add_argument("--c-reduce", default="max", choices=["max", "mean", "median"],
                    help="How to collapse channels when --c-index is not provided.")
    ap.add_argument("--out", default="outputs/roi", help="Output folder for ROI zips & masks")
    args = ap.parse_args()

    stack = open_stack_any(args.input, axes=args.axes,
                           zproject=args.zproject, c_index=args.c_index, c_reduce=args.c_reduce)  # (T,Y,X)

    T, H, W = stack.shape
    os.makedirs(args.out, exist_ok=True)
    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    v = napari.Viewer(title="Neuron Annotator — polygons/freehand (ImageJ-compatible)")
    try:
        v.add_image(stack, name="img", interpolation2d="nearest")
    except TypeError:
        v.add_image(stack, name="img", interpolation="nearest")

    # --- ROIs layer ---
    shapes = v.add_shapes(
        name="ROIs",
        ndim=3,
        shape_type="polygon",           # polygon / freehand path via toolbuttons
        edge_color="#FFFF00",
        edge_width=0.5,
        face_color=[0, 0, 0, 0],
    )
    # draw polygons by default (finish with double-click or Enter)
    try: shapes.mode = 'add_polygon'
    except Exception: pass

    # direct color mode so our edges always show
    try:
        shapes.edge_color_mode = 'direct'
        shapes.face_color_mode = 'direct'
    except Exception:
        pass

    # defaults for NEW shapes
    shapes.current_edge_color = "#FFFF00"
    shapes.current_edge_width = 0.5
    shapes.current_face_color = [0, 0, 0, 0]

    # features table (time + label)
    shapes.features = pd.DataFrame({"t": pd.Series(dtype="int64"),
                                    "label": pd.Series(dtype="object")})

    # on-screen numbering (tiny)
    # try:
    #     shapes.text = {"text": "{label}", "size": 8, "color": "yellow", "anchor": "center"}
    # except Exception:
    #     shapes.text = {"string": "{label}", "size": 8, "color": "yellow", "anchor": "center"}

    # ---- style helper ----
    def _style_all(width=0.5, color="#FFFF00", alpha=0.0):
        """Apply bright thin edges + transparent fill. Use SCALARS, not arrays."""
        # force direct color mode so per-shape scalars are honored
        try:
            shapes.edge_color_mode = "direct"
            shapes.face_color_mode = "direct"
        except Exception:
            pass

        # --- scalars for everyone (avoids napari broadcasting bug) ---
        shapes.edge_width = float(width)        # <- SCALAR, not array
        shapes.edge_color = color               # <- single color string
        shapes.face_color = [0, 0, 0, alpha]    # transparent fill (RGBA)

        # defaults for NEW shapes
        shapes.current_edge_width = float(width)
        shapes.current_edge_color = color
        shapes.current_face_color = [0, 0, 0, alpha]

    _style_all()

   # ---- utility: shape validity (prevents labels on single clicks) ----
    def _poly_area(pts):
        # This function now correctly handles both 2D (Y,X) and 3D (T,Y,X) coordinates
        # by always using the last two columns for the area calculation.
        y = pts[:, -2]  # Y coordinate is the second to last column
        x = pts[:, -1]  # X coordinate is the last column
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _is_valid_shape(i, min_area=2.0):
        # --- ADD THIS SAFETY CHECK ---
        # First, ensure the index 'i' is valid for the shapes.data list
        if i >= len(shapes.data):
            return False
        # --- END OF CHECK ---

        pts = np.asarray(shapes.data[i])
        return (pts.shape[0] >= 3) and (_poly_area(pts) >= min_area)
    


    # ---- Sync .properties for older napari (prevents 'None') ----
    def _sync_props():
        try:
            props = shapes.features.to_dict("list")
            props = {k: np.array(v, dtype=object) for k, v in props.items()}
            shapes.properties = props
        except Exception:
            pass

    # ---- Auto-number 1..N only for valid shapes on a frame ----
    def _renumber_frame(t: int):
        """
        A robust function to re-number all valid shapes on a specific time frame.
        It clears all old labels for the frame and assigns new ones starting from 1.
        """
        # Exit if the features table is not ready
        if 't' not in shapes.features.columns or len(shapes.features) == 0:
            return

        all_features = shapes.features
        
        # --- 1. Find all shapes belonging to the current time frame 't' ---
        indices_on_this_frame = all_features.index[all_features['t'] == t].tolist()

        # --- 2. From that list, find which ones are actual valid polygons ---
        valid_indices = []
        for i in indices_on_this_frame:
            if _is_valid_shape(i):
                valid_indices.append(i)
        
        # --- 3. Clear the labels for ALL shapes on this frame (valid or not) ---
        if indices_on_this_frame: # only if there are any shapes on this frame
            all_features.loc[indices_on_this_frame, 'label'] = ''

        # --- 4. Assign fresh labels (1, 2, 3...) ONLY to the valid shapes ---
        for k, index in enumerate(valid_indices):
            all_features.at[index, 'label'] = str(k + 1)
        
        # --- 5. Update the layer with the modified features and refresh the view ---
        shapes.features = all_features
        _sync_props()
        try:
            shapes.refresh_text()
        except AttributeError:
            pass
        
        # --- ADD THIS NEW BLOCK AT THE END ---
        # 6. Update the status bar with the neuron count for the current frame
        count = len(valid_indices)
        v.status = f"Neurons on frame {t}: {count}"
        # --- END OF NEW BLOCK ---
    
    

    # ---- respond to drawing/editing ----
    def _tag_new_shapes(event=None):
        """Reacts to data changes by tagging new shapes with the current time."""
        current_num_shapes = len(shapes.data)
        current_num_features = len(shapes.features)
        
        t = int(v.dims.current_step[0])

        if current_num_shapes > current_num_features:
            # This means shapes were ADDED
            diff = current_num_shapes - current_num_features
            
            with shapes.events.data.blocker():
                current_data = list(shapes.data)
                start_index = len(current_data) - diff
                for i in range(start_index, len(current_data)):
                    shape_2d = current_data[i]
                    if shape_2d.shape[1] == 2: # Only upgrade if it's a 2D shape
                        t_column = np.full((len(shape_2d), 1), t)
                        shape_3d = np.hstack([t_column, shape_2d])
                        current_data[i] = shape_3d
                shapes.data = current_data

            # Add new rows to the features table
            new_features = pd.DataFrame({'t': [t] * diff, 'label': [''] * diff})
            shapes.features = pd.concat([shapes.features, new_features], ignore_index=True)

        # Always re-number the current frame after any potential change
        _renumber_frame(t)

    # -------- hotkeys --------
    
    @v.bind_key("e")       # edit existing shapes
    def _edit_mode(_):
        try: shapes.mode = "select"
        except Exception: pass

    @v.bind_key("p")       # draw new polygons
    def _draw_poly(_):
        try: shapes.mode = "add_polygon"
        except Exception: pass

    @v.bind_key("h")       # freehand (path)
    def _draw_freehand(_):
        try: shapes.mode = "add_path"
        except Exception: pass

    @v.bind_key("n")
    def _next(_): v.dims.set_current_step(0, min(int(v.dims.current_step[0]) + 1, T - 1))

    @v.bind_key("b")
    def _back(_): v.dims.set_current_step(0, max(int(v.dims.current_step[0]) - 1, 0))

    # Undo last FINISHED ROI on this frame
    @v.bind_key("u")
    def _undo_last(_):
        t = int(v.dims.current_step[0])
        
        # Get the master indices of all valid shapes on the current frame
        indices_on_this_frame = [
            i for i, t_val in enumerate(shapes.features['t']) 
            if t_val == t and _is_valid_shape(i)
        ]
        
        if not indices_on_this_frame:
            return # No valid shapes to undo on this frame

        # Find the index of the last shape to remove
        index_to_remove = indices_on_this_frame[-1]
        
        # Use napari's own safe removal method
        shapes.selected_data = {index_to_remove}
        shapes.remove_selected()
        
        # Napari's remove_selected will trigger the shapes.events.data event,
        # which will call _tag_new_shapes. However, that function doesn't handle
        # deletions well, so we need to manually sync the features table here.
        
        # Manually re-sync the features table to match the data list
        # This is a robust way to handle the aftermath of a deletion
        if len(shapes.data) < len(shapes.features):
             shapes.features = shapes.features.iloc[list(shapes.data_to_features.values())].reset_index(drop=True)
        
        _renumber_frame(t) # Explicitly re-number after the undo
    @v.bind_key("s")
    def _save_zip(_):
        t = int(v.dims.current_step[0])
        path = os.path.join(args.out, f"rois_t{t:04d}.zip")
        save_roizip_for_frame(path, shapes, t, stroke_color=(255,255,0), stroke_width=1.4)
        print("✅ saved", path)

    @v.bind_key("m")
    def _save_mask(_):
        t = int(v.dims.current_step[0])
        mask = mask_for_frame(H, W, shapes, t)
        path = os.path.join(mask_dir, f"mask_t{t:04d}.tif")
        imwrite(path, mask, dtype=np.uint8)
        print("✅ saved", path)

    @v.bind_key("Control-Shift-S")
    def _save_all(_):
        if len(shapes.features) == 0 or "t" not in shapes.features or shapes.features["t"].isna().all():
            print("No annotations to save.")
            return
        
        frames_to_save = sorted(set(int(t) for t in shapes.features["t"].dropna()))
        print(f"Found annotations on {len(frames_to_save)} frames. Saving now...")

        image_dir = os.path.join(args.out, "images")
        mask_dir = os.path.join(args.out, "masks")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # --- NEW: Scan directory to find the highest existing number ---
        existing_indices = []
        for f in os.listdir(image_dir):
            if f.startswith('image_') and f.endswith('.tif'):
                try:
                    num = int(f.replace('image_', '').replace('.tif', ''))
                    existing_indices.append(num)
                except ValueError:
                    continue
        start_num = max(existing_indices) + 1 if existing_indices else 1
        print(f"Starting file numbering at {start_num:04d}...")
        # --- END NEW ---

        for i, t in enumerate(frames_to_save):
            current_num = start_num + i
            
            mask_path = os.path.join(mask_dir, f"mask_{current_num:04d}.tif")
            image_path = os.path.join(image_dir, f"image_{current_num:04d}.tif")
            
            mask = mask_for_frame(H, W, shapes, t)
            imwrite(mask_path, mask, dtype=np.uint16)
            
            image_frame = stack[t]
            if hasattr(image_frame, 'compute'): image_frame = image_frame.compute()
            imwrite(image_path, image_frame)
            
        print(f"✅ Saved data for {len(frames_to_save)} frames to '{args.out}' folder.")

        
    @v.bind_key("l")
    def _toggle_labels(_):
        try:
            shapes.text.visible = not getattr(shapes.text, "visible", True)
        except Exception:
            if isinstance(shapes.text, dict):
                key = "text" if "text" in shapes.text else "string"
                shapes.text[key] = "" if shapes.text.get(key) else "{label}"

    # ... (all your @v.bind_key functions are here) ...
    @v.bind_key("f")
    def _toggle_fill(_):
        _faint["on"] = not _faint["on"]
        alpha = 0.12 if _faint["on"] else 0.0
        rgba = [1, 1, 0, alpha]
        shapes.face_color = rgba
        shapes.current_face_color = rgba

# ---- EVENT CALLBACKS ----
    #
    # THIS IS THE CRUCIAL MISSING LINE:
    # Connect the main logic to the event that fires when shapes data changes.
    shapes.events.data.connect(_tag_new_shapes)
    #
    #

    # Add a callback to re-number when the time slider is moved
    @v.dims.events.current_step.connect
    def _on_frame_change(event):
        # event.value is a tuple of the current step for all dimensions
        current_frame = event.value[0]  # Assuming Time is the first dimension
        _renumber_frame(current_frame)

    # Trigger a renumber for the initial frame on startup
    _renumber_frame(0)

    # small status hint
    v.status = "Draw with Polygon (P) or Freehand (H). Finish with double-click or Enter. Esc cancels current draw. U = undo last."

    napari.run()


if __name__ == "__main__":
    main()

