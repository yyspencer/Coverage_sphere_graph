# MoCap Coverage Pipeline
This README explains the four-file workflow used to convert CMU SMPL-X motion data into a unit-sphere MoCap coverage graph and compare it with another dataset such as PSUTMM-100.

## 1. Pipeline Overview
The overall workflow is:

```text
CMU *_stageii.npz
        |
        v
extract_cmu.py
        |
        v
*_smplx_output.npz
joints: (T, J, 3)
        |
        v
extract_to_sphere.py
        |
        v
.npy / .npz
pose: (4, J, T)
        |
        v
mocap_unit_sphere.py
        |
        v
unit-sphere limb-orientation coverage graph

```
`npzkey.py` is an optional inspection/debugging utility and is not a required transformation stage.

## 2. File Overview

### `extract_cmu.py`
`extract_cmu.py` converts the original CMU `*_stageii.npz` SMPL-X files into explicit 3D joint coordinates.
The input files contain SMPL-X parameters such as body pose, root orientation, translation, shape coefficients, jaw pose, eye pose, and gender.
The script loads the corresponding SMPL-X body model, performs the SMPL-X forward pass, and extracts joint positions for each frame.
Frames are processed in batches, and CUDA is used automatically when PyTorch detects an available GPU.
The important outputs are `joints` with shape `(T, J, 3)` and `global_orient` with shape `(T, 3)`.
Conceptually, this step performs:
`SMPL-X parameters -> SMPL-X body model -> 3D joint coordinates`

### `extract_to_sphere.py`
`extract_to_sphere.py` reformats the output from `extract_cmu.py` into the array layout expected by `mocap_unit_sphere.py`.
The input `joints` array has shape `(T, J, 3)`.
The script transposes it to `(3, J, T)` and then appends a confidence channel filled with `1.0`.
The final output has shape `(4, J, T)` where the four channels are X, Y, Z, and confidence.
The file can be saved as `.npy` or `.npz`; `.npy` is the simplest choice for this workflow.

### `mocap_unit_sphere.py`
`mocap_unit_sphere.py` creates the final limb-orientation trajectory graph.
The script selects a proximal joint and a distal joint, such as `R_Elbow` and `R_Wrist`.
For every frame it calculates `V = distal - proximal`, then normalizes the vector with `U = V / ||V||`.
Because `U` has unit length, each valid point lies on the unit sphere.
Connecting these points over time produces a trajectory that represents how the selected limb changes orientation.
This graph does not show where the wrist or elbow travels through the room; it shows the direction of the limb segment.
The representation removes absolute translation and limb length.
The script supports `.mat`, `.npy`, and `.npz` input and handles both the legacy 17-joint skeleton and the larger SMPL-X joint representation.

### `npzkey.py`
`npzkey.py` is a small debugging script used to inspect `.npz` files.
It prints the available keys, the shape of each stored array, and the dtype.
It is useful after `extract_cmu.py` when you want to confirm that the generated file contains a `joints` array with the expected shape.

## 3. Requirements
Main Python packages:
`numpy`, `torch`, `smplx`, `tqdm`, `matplotlib`, `scipy`, and `h5py`.
Install them with:

```bash
pip install numpy torch smplx tqdm matplotlib scipy h5py

```
If you are using an NVIDIA GPU, install the CUDA-enabled PyTorch build appropriate for your system.

## 4. SMPL-X Model Files
`extract_cmu.py` expects the SMPL-X body model files under `body_models/smplx/`.
A typical layout is:

```text
body_models/
└── smplx/
    ├── SMPLX_MALE.npz
    ├── SMPLX_FEMALE.npz
    └── ...

```
The script reads the `gender` field from each CMU motion file and selects the corresponding body model.
If your SMPL-X models are stored somewhere else, modify the model path in `extract_cmu.py`.

## 5. Expected CMU Input
The CMU input directory should contain files ending in `*_stageii.npz`.
For example:

```text
CMU_35/
├── 35_01_stageii.npz
├── 35_02_stageii.npz
├── 35_03_stageii.npz
└── ...

```

## 6. Step 1 — Extract CMU 3D Joints
Run:

```bash
python extract_cmu.py --input_folder <CMU_STAGEII_FOLDER> --output_dir <SMPLX_OUTPUT_FOLDER>

```
Example:

```bash
python extract_cmu.py --input_folder E:/LPAC/RoM/CMU_35 --output_dir E:/LPAC/RoM/CMU_35_EXTRACTED

```
Optional arguments are `--batch_size` and `--num_betas`.
The current defaults are `batch_size=128` and `num_betas=16`.
An explicit example is:

```bash
python extract_cmu.py --input_folder E:/LPAC/RoM/CMU_35 --output_dir E:/LPAC/RoM/CMU_35_EXTRACTED --batch_size 128 --num_betas 16

```
The script searches the specified folder for `*_stageii.npz` files and processes them one at a time.
The resulting output should contain `joints` and `global_orient`.
The key array for the next stage is:
`joints.shape = (T, J, 3)`

## 7. Optional — Inspect the Extracted NPZ
If you want to verify an intermediate file, open `npzkey.py` and set:

```python
PATH = "<PATH_TO_EXTRACTED_FILE.npz>"

```
Then run:

```bash
python npzkey.py

```
Confirm that the file contains a `joints` key and that the shape is `(T, J, 3)`.
If the shape or keys are unexpected, resolve that before continuing.

## 8. Step 2 — Convert to Sphere-Input Format
For one extracted file, run:

```bash
python extract_to_sphere.py --in_file <SMPLX_OUTPUT_FILE.npz> --out_dir <SPHERE_INPUT_FOLDER>

```
Example:

```bash
python extract_to_sphere.py --in_file E:/LPAC/RoM/CMU_35_EXTRACTED/35_33_stageii_smplx_output.npz --out_dir E:/LPAC/RoM/CMU_35_SPHERE

```
To convert all `.npz` files in a directory recursively, run:

```bash
python extract_to_sphere.py --in_dir <SMPLX_OUTPUT_FOLDER> --out_dir <SPHERE_INPUT_FOLDER> --ext npy

```
Example:

```bash
python extract_to_sphere.py --in_dir E:/LPAC/RoM/CMU_35_EXTRACTED --out_dir E:/LPAC/RoM/CMU_35_SPHERE --ext npy

```
Use `--ext npy` for `.npy` output or `--ext npz` for `.npz` output.
The converted data should have shape `(4, J, T)`.

## 9. Step 3 — Choose the Limb
The unit-sphere graph analyzes one joint-to-joint segment at a time.
Use `--prox` for the proximal joint and `--dist` for the distal joint.
Common examples are:
- `R_Shoulder -> R_Elbow`: right upper arm
- `R_Elbow -> R_Wrist`: right forearm
- `L_Shoulder -> L_Elbow`: left upper arm
- `L_Elbow -> L_Wrist`: left forearm
- `R_Hip -> R_Knee`: right thigh
- `R_Knee -> R_Ankle`: right lower leg
- `L_Hip -> L_Knee`: left thigh
- `L_Knee -> L_Ankle`: left lower leg
For example, `--prox R_Elbow --dist R_Wrist` means that the plot represents right-forearm orientation.

## 10. Step 4 — Generate the Coverage Graph
General command:

```bash
python mocap_unit_sphere.py --files <FILE_1> <FILE_2> --prox <PROXIMAL_JOINT> --dist <DISTAL_JOINT>

```
Example comparing a PSU motion with a converted CMU motion:

```bash
python mocap_unit_sphere.py --files PSU_MOTION.npy CMU_MOTION.npy --prox R_Elbow --dist R_Wrist

```
The script loads both files, resolves the joint indices, computes the normalized limb vectors, and overlays the resulting trajectories on the unit sphere.
Each plotted point represents `normalized(distal_joint - proximal_joint)` for one frame.

## 11. Fixed Sample Count
Different recordings may contain very different numbers of frames.
Use `--samples 600` when comparing trajectories so one long recording does not simply look denser because it contains more frames.
Example:

```bash
python mocap_unit_sphere.py --files PSU_MOTION.npy CMU_MOTION.npy --prox R_Elbow --dist R_Wrist --samples 600

```
The current implementation uses uniform temporal subsampling.
It selects approximately evenly spaced existing frames; it does not interpolate a new trajectory.
Use `--samples 0` to disable this behavior.

## 12. Confidence Threshold
The optional `--conf_thresh` argument can be used to reject frames where either selected joint has low confidence.
Example:
`--conf_thresh 0.5`
Converted CMU files currently receive confidence `1.0` for every joint and frame, so this threshold normally has no effect on them.
It can still matter for other MoCap or pose-estimation files that contain real confidence values.

## 13. Save the Graph
Use `--save` to write the figure to disk.
Example:

```bash
python mocap_unit_sphere.py --files PSU_MOTION.npy CMU_MOTION.npy --prox R_Elbow --dist R_Wrist --samples 600 --save right_forearm_coverage.png

```
Matplotlib can also save formats such as PDF.

## 14. Custom Plot Title
Use the optional `--title` argument.
Example:

```bash
python mocap_unit_sphere.py --files PSU_MOTION.npy CMU_MOTION.npy --prox R_Elbow --dist R_Wrist --samples 600 --title "Right Forearm Unit-Sphere Coverage" --save right_forearm_coverage.png

```

## 15. Joint Naming
The compact legacy representation uses names such as `R_Shoulder`, `R_Elbow`, `R_Wrist`, `L_Shoulder`, `L_Elbow`, `L_Wrist`, `R_Hip`, `R_Knee`, and `R_Ankle`.
The SMPL-X representation uses names such as `right_shoulder`, `right_elbow`, `right_wrist`, `left_shoulder`, `left_elbow`, and `left_wrist`.
`mocap_unit_sphere.py` contains a mapping between the supported legacy names and the corresponding SMPL-X names.
This lets a command such as `--prox R_Elbow --dist R_Wrist` work for both supported dataset representations.

## 16. How to Read the Graph
For `R_Elbow -> R_Wrist`, the plotted quantity is:
`(Wrist - Elbow) / ||Wrist - Elbow||`
Therefore the graph represents right-forearm orientation over time.
It is not the absolute wrist trajectory.
If a trajectory visits a particular region of the sphere, the corresponding motion contains frames where the selected limb points in those directions.
Comparing trajectories provides a qualitative view of how similar or different two motions are in limb-orientation space.
A trajectory that reaches regions not covered by the other motion indicates additional visible orientation coverage.

## 17. Important Limitation
The unit-vector representation removes global translation and limb length.
It does not automatically remove global body rotation.
Two subjects performing similar poses while facing different global directions may therefore create rotated trajectories.
If the pipeline is later used for quantitative dataset-level comparisons, consider transforming the skeleton or limb vectors into a body-relative or root-relative coordinate system before calculating coverage.

## 18. End-to-End Workflow
The normal workflow is:
1. Obtain the CMU `*_stageii.npz` files.
2. Make sure the SMPL-X body models are available.
3. Run `extract_cmu.py`.
4. Optionally inspect the output with `npzkey.py`.
5. Run `extract_to_sphere.py`.
6. Choose the limb you want to analyze.
7. Select the CMU and comparison MoCap files.
8. Run `mocap_unit_sphere.py`.
9. Inspect or save the coverage graph.
The minimal command sequence is:

```bash
python extract_cmu.py --input_folder CMU_INPUT --output_dir CMU_EXTRACTED
python extract_to_sphere.py --in_dir CMU_EXTRACTED --out_dir CMU_SPHERE --ext npy
python mocap_unit_sphere.py --files PSU_MOTION.npy CMU_MOTION.npy --prox R_Elbow --dist R_Wrist --samples 600 --save mocap_coverage.png

```

## 19. Data Shape Summary
The most important shape transitions are:
`CMU stageii file -> SMPL-X parameters`
`extract_cmu.py -> joints = (T, J, 3)`
`extract_to_sphere.py -> pose = (4, J, T)`
`mocap_unit_sphere.py -> xyz = (3, J, T)`
For one selected limb:
`P = proximal positions = (T, 3)`
`D = distal positions = (T, 3)`
`V = D - P = (T, 3)`
`U = V / ||V|| = (T, 3)`
Every valid row of `U` has approximately unit length and therefore lies on the unit sphere.

## 20. Basic Sanity Checks
Before interpreting a graph, confirm:
- `extract_cmu.py` completed without errors.
- the extracted `.npz` contains the `joints` key.
- the `joints` array has shape `(T, J, 3)`.
- the converted file has shape `(4, J, T)`.
- the requested proximal and distal joints are supported.
- the selected files correspond to the motions you actually intend to compare.
If the graph looks clearly wrong, check the array shapes and joint mapping before changing the visualization logic.

## 21. Current Scope
The current pipeline is primarily a qualitative limb-orientation coverage visualization.
It can help answer questions such as:
- Which directions does this limb reach?
- How broad is the visible orientation coverage?
- Do two motions occupy similar regions of orientation space?
The trajectory graph alone is not yet a quantitative dataset-diversity metric.
Possible future extensions include spherical occupancy, angular coverage, geodesic distance, spherical KDE, coverage percentage, entropy, trajectory similarity, and root-relative orientation normalization.
These can all be built on the same normalized limb-vector representation already produced by `mocap_unit_sphere.py`.

## 22. Short Reference
`extract_cmu.py`: SMPL-X parameters -> 3D joints.
`extract_to_sphere.py`: `(T, J, 3)` -> `(4, J, T)`.
`mocap_unit_sphere.py`: limb pair -> normalized orientation trajectory -> sphere graph.
`npzkey.py`: inspect `.npz` keys, shapes, and dtypes.
Complete conceptual pipeline:
`CMU SMPL-X -> 3D joints -> common MoCap format -> limb vectors -> unit vectors -> unit-sphere coverage visualization`
