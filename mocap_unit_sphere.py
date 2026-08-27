#!/usr/bin/env python3
"""
Unit-sphere limb-orientation trajectories from MoCap files.

- Inputs: 2+ files (.mat v7.3 HDF5 or older .mat, or .npy/.npz)
- Expected data shape: (4, n_joints, n_frames) where 4 = x,y,z,conf
- Plots: time-ordered trajectory lines on a unit sphere for each file (no time coloring)

Example:
  python unit_sphere_trajectory.py \
    --files MOCAP_3D_2.mat motionB.npy motionC.mat \
    --prox R_Elbow --dist R_Wrist \
    --samples 600 --conf_thresh 0.0 \
    --title "Unit Sphere Projection (Limb Orientation)"

Notes:
- To avoid "longer motion looks richer", set --samples to a fixed value (e.g., 600).
- Confidence is optional: if present, frames below threshold are treated as gaps (line breaks).
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional: scipy for old MAT (non-v7.3). We will fall back gracefully.
try:
    from scipy.io import loadmat  # type: ignore
except Exception:
    loadmat = None

# Optional: h5py for v7.3 MAT
try:
    import h5py  # type: ignore
except Exception:
    h5py = None


JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

JOINT_ORDER = [
    "R_Shoulder", "R_Elbow", "R_Wrist",
    "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Hip", "R_Knee", "R_Ankle",
    "L_Hip", "L_Knee", "L_Ankle",
    "Pelvis", "Waist", "NeckTop", "Clavicle", "Thorax"
]
JOINTS = {name: i for i, name in enumerate(JOINT_ORDER)}

# Add this near the top, after JOINT_NAMES / JOINTS are defined:

JOINT_NAMES_MAP = {name: i for i, name in enumerate(JOINT_NAMES)}

# Minimal translation from JOINT_ORDER-style args -> SMPL-X/OpenPose-style names in JOINT_NAMES
ORDER_TO_NAMES = {
    "R_Shoulder": "right_shoulder",
    "R_Elbow": "right_elbow",
    "R_Wrist": "right_wrist",
    "L_Shoulder": "left_shoulder",
    "L_Elbow": "left_elbow",
    "L_Wrist": "left_wrist",
    "R_Hip": "right_hip",
    "R_Knee": "right_knee",
    "R_Ankle": "right_ankle",
    "L_Hip": "left_hip",
    "L_Knee": "left_knee",
    "L_Ankle": "left_ankle",
    "Pelvis": "pelvis",
    # If you ever use these, pick the closest matches:
    "Clavicle": None,  # could be left_collar/right_collar depending on intent
    "Thorax": None,    # could be spine3 / thorax if present in your naming
    "NeckTop": "neck",   # could be neck / head depending on your naming
    "Waist": None,     # could be spine1/spine2 depending on definition
}

def resolve_joint_index(joint_name: str, n_joints: int) -> int:
    """
    If n_joints == 17: use JOINTS (JOINT_ORDER names like R_Elbow)
    Else: use JOINT_NAMES (127+ names like right_elbow), with fallback translation.
    """
    # Case 1: 17-joint input (your legacy compact format)
    if n_joints == 17:
        idx = JOINTS.get(joint_name)
        if idx is None:
            raise ValueError(
                f"J=17 input expects JOINT_ORDER names. Got '{joint_name}'. "
                f"Valid: {list(JOINTS.keys())}"
            )
        return idx

    # Case 2: 127+ joint input
    # Accept direct JOINT_NAMES keys (case-insensitive)
    key = joint_name.strip()
    key_l = key.lower()
    if key_l in JOINT_NAMES_MAP:
        return JOINT_NAMES_MAP[key_l]

    # Accept JOINT_ORDER-style names by translating to JOINT_NAMES style
    mapped = ORDER_TO_NAMES.get(key)
    if mapped is None:
        raise ValueError(
            f"J={n_joints} input: could not resolve '{joint_name}'. "
            f"Try passing a JOINT_NAMES entry (e.g., 'right_elbow'), or extend ORDER_TO_NAMES."
        )
    return JOINT_NAMES_MAP[mapped]

def _find_pose_array_in_mat_dict(d: dict) -> np.ndarray:
    """Find an array shaped (4, n_joints, n_frames) in a scipy loadmat dict."""
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[0] == 4:
            return v
    raise ValueError("Could not find a (4, n_joints, n_frames) array in .mat file.")


def load_pose(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load pose from .mat/.npy/.npz.

    Accepts these layouts:
      - (4, J, T) or (3, J, T)
      - (T, J, 4) or (T, J, 3)

    Returns:
      XYZ:  (3, J, T)
      CONF: (J, T) or None
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext == ".npz":
        z = np.load(path, allow_pickle=False)
        for key in ["POSE", "pose", "data", "arr"]:
            if key in z:
                arr = z[key]
                break
        else:
            arr = z[z.files[0]]
    elif ext == ".mat":
        # Try v7.3 via h5py first
        arr = None
        if h5py is not None:
            try:
                with h5py.File(path, "r") as f:
                    if "POSE" in f and hasattr(f["POSE"], "shape"):
                        arr = np.array(f["POSE"])
                    else:
                        for k in f.keys():
                            obj = f[k]
                            if hasattr(obj, "shape") and len(obj.shape) == 3 and obj.shape[0] in (3, 4):
                                arr = np.array(obj)
                                break
                # If not HDF5, this raises OSError and we fall back
            except OSError:
                arr = None

        if arr is None:
            if loadmat is None:
                raise RuntimeError("scipy not available to read non-v7.3 .mat files.")
            d = loadmat(path)
            # Prefer 'POSE' key if present
            if "POSE" in d and isinstance(d["POSE"], np.ndarray):
                arr = d["POSE"]
            else:
                arr = _find_pose_array_in_mat_dict(d)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .mat/.npy/.npz")

    arr = np.asarray(arr)
    if arr.ndim != 3:
        print(arr)
        raise ValueError(f"{path}: expected 3D array, got {arr.shape}")

    # Normalize layout to (C, J, T) where C in {3,4}
    # Case A: already (C, J, T)
    if arr.shape[0] in (3, 4) and arr.shape[1] >= 1:
        c, j, t = arr.shape
        arr_cjt = arr

    # Case B: (T, J, C)
    elif arr.shape[2] in (3, 4) and arr.shape[1] >= 1:
        t, j, c = arr.shape
        arr_cjt = np.transpose(arr, (2, 1, 0))  # (C, J, T)

    else:
        raise ValueError(
            f"{path}: unrecognized layout. Got {arr.shape}. "
            "Expected (4,J,T)/(3,J,T) or (T,J,4)/(T,J,3)."
        )

    # Split XYZ/CONF
    if arr_cjt.shape[0] == 4:
        xyz = arr_cjt[:3].astype(float)
        conf = arr_cjt[3].astype(float)  # (J, T)
    else:
        xyz = arr_cjt[:3].astype(float)
        conf = None

    return xyz, conf


def unit_sphere_trajectory(
    xyz: np.ndarray,
    prox_idx: int,
    dist_idx: int,
    conf: Optional[np.ndarray] = None,
    conf_thresh: float = 0.0,
    samples: Optional[int] = 600,
) -> np.ndarray:
    """
    Build time-ordered unit vectors (trajectory) from proximal->distal segment.

    Returns:
      U: (T', 3) with NaNs inserted where invalid/gaps occur (line breaks).
    """
    # xyz: (3, J, T)
    if xyz.shape[0] != 3:
        raise ValueError("xyz must have shape (3, n_joints, n_frames)")

    P = xyz[:, prox_idx, :].T  # (T,3)
    D = xyz[:, dist_idx, :].T  # (T,3)
    V = D - P
    n = np.linalg.norm(V, axis=1, keepdims=True)

    valid = np.isfinite(n.squeeze()) & (n.squeeze() > 1e-8)
    if conf is not None:
        c_ok = (conf[prox_idx, :] >= conf_thresh) & (conf[dist_idx, :] >= conf_thresh)
        valid = valid & c_ok

    U = np.full_like(V, np.nan, dtype=float)
    U[valid] = V[valid] / n[valid]

    # Resample to fixed number of points to avoid duration bias in visualization
    if samples is not None and samples > 1 and U.shape[0] > samples:
        idx = np.linspace(0, U.shape[0] - 1, samples).astype(int)
        U = U[idx]

    return U


def plot_unit_sphere(ax):
    phi = np.linspace(0, np.pi, 25)
    theta = np.linspace(0, 2 * np.pi, 50)
    X = np.outer(np.sin(phi), np.cos(theta))
    Y = np.outer(np.sin(phi), np.sin(theta))
    Z = np.outer(np.cos(phi), np.ones_like(theta))
    ax.plot_wireframe(X, Y, Z, linewidth=0.35, alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="2+ input files: .mat/.npy/.npz")
    parser.add_argument("--prox", required=True, help=f"Proximal joint name (e.g. R_Elbow). Options: {list(JOINTS.keys())}")
    parser.add_argument("--dist", required=True, help=f"Distal joint name (e.g. R_Wrist). Options: {list(JOINTS.keys())}")
    parser.add_argument("--samples", type=int, default=600, help="Fixed #points per trajectory (reduces duration bias). Use 0 to disable.")
    parser.add_argument("--conf_thresh", type=float, default=0.0, help="Confidence threshold; below breaks trajectory.")
    parser.add_argument("--title", type=str, default="Unit Sphere Projection (Limb Orientation)\nTime-ordered trajectory (no time coloring)")
    parser.add_argument("--save", type=str, default="", help="Optional output image path (png/pdf).")
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    samples = args.samples if args.samples and args.samples > 0 else None

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_unit_sphere(ax)

    for path in args.files:
        xyz, conf = load_pose(path)

        n_joints = xyz.shape[1]
        prox_idx = resolve_joint_index(args.prox, n_joints)
        dist_idx = resolve_joint_index(args.dist, n_joints)

        U = unit_sphere_trajectory(
            xyz=xyz,
            prox_idx=prox_idx,
            dist_idx=dist_idx,
            conf=conf,
            conf_thresh=args.conf_thresh,
            samples=samples,
        )
        if 'Subject' in path:
            label = 'PSUTMM-100 Subject 7 Take 2'
        else:
            label = "CMU Subject 35 Take x"

        # Decide style by which "group" this path belongs to
        # (you can change this condition to whatever reliably distinguishes them)
        is_psu = ("Subject" in path) or ("PSU" in path)

        if is_psu:
            ax.plot(U[:, 0], U[:, 1], U[:, 2],
                    linewidth=3.5, color="C1", alpha=0.25,
                    label=label)
        else:
            ax.plot(U[:, 0], U[:, 1], U[:, 2],
                    linewidth=3.5, color="C0", alpha=1,
                    label=label)


    ax.set_title(args.title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
