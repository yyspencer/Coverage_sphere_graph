# preprocessing script to get the output of extract_cmu & fill it to the format for mocap unit sphere. 
#!/usr/bin/env python3
"""
Convert extract_cmu.py output .npz (with joints shaped (N, J, 3))
into mocap_unit_sphere.py input format: a single array shaped (4, J, N).

- xyz channels: x,y,z
- confidence channel: all ones
- output: .npy by default (recommended), or .npz containing a key "POSE"
"""

import os
import argparse
import numpy as np


def convert_one(npz_path: str, out_path: str, save_as: str = "npy") -> None:
    data = np.load(npz_path, allow_pickle=False)

    if "joints" not in data:
        raise KeyError(f"{npz_path} does not contain key 'joints'. Keys: {list(data.keys())}")

    joints = data["joints"]  # expected (N, J, 3)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"{npz_path}: expected joints shape (N, J, 3), got {joints.shape}")

    N, J, _ = joints.shape

    # (N, J, 3) -> (3, J, N)
    xyz = np.transpose(joints, (2, 1, 0)).astype(np.float32)

    # confidence (J, N) all ones -> (1, J, N)
    conf = np.ones((1, J, N), dtype=np.float32)

    # stack -> (4, J, N)
    pose = np.concatenate([xyz, conf], axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if save_as == "npy":
        # mocap_unit_sphere.py loads .npy directly and accepts (C, J, T)
        np.save(out_path, pose)
    elif save_as == "npz":
        # mocap_unit_sphere.py also supports .npz; it looks for POSE/pose/data/arr
        np.savez_compressed(out_path, POSE=pose)
    else:
        raise ValueError("save_as must be 'npy' or 'npz'")

    print(f"[OK] {npz_path} -> {out_path} | pose shape={pose.shape} (4,J,N) where J={J}, N={N}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, help="Single input .npz file to convert")
    parser.add_argument("--in_dir", type=str, help="Folder of .npz files to convert (recursive)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder")
    parser.add_argument("--ext", type=str, default="npy", choices=["npy", "npz"], help="Output format")
    args = parser.parse_args()

    if not args.in_file and not args.in_dir:
        raise ValueError("Provide either --in_file or --in_dir")

    inputs = []
    if args.in_file:
        inputs.append(args.in_file)
    if args.in_dir:
        for root, _, files in os.walk(args.in_dir):
            for f in files:
                if f.lower().endswith(".npz"):
                    inputs.append(os.path.join(root, f))

    if not inputs:
        raise ValueError("No .npz files found.")

    for npz_path in inputs:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        out_path = os.path.join(args.out_dir, f"{base}.{args.ext}")
        convert_one(npz_path, out_path, save_as=args.ext)


if __name__ == "__main__":
    main()
