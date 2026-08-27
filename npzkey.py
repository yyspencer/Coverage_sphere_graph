import numpy as np

#PATH = "E:/LPAC/GroundLink/PSUTMM100/Subject7/Subject7_MOCAP_MRK_2_gt_stageii.npz"
PATH = "E:/LPAC/RoM/CMU_35/35_33_stageii/35_33_stageii_smplx_output.npz"

z = np.load(PATH, allow_pickle=True)   # z is an NpzFile (dict-like)

print("Keys:", z.files)                # or: list(z.keys())

# Inspect each array
for k in z.files:
    arr = z[k]
    print(f"{k:>12}  shape={arr.shape}  dtype={arr.dtype}")