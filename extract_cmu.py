import argparse
from tqdm import tqdm
import os
import numpy as np
import torch
from smplx import SMPLX
from glob import glob

def load_data(smplx_file, batch_size=1, num_betas=16):
    """Load SMPLX and other data for a subject and take."""
    smplx_data = np.load(smplx_file)
    # Extract gender from the array(gender, dtype='<U6')
    gender = smplx_data['gender'].item()
    model_path = f'body_models/smplx/SMPLX_{gender.upper()}.npz'
    model = SMPLX(model_path=model_path, gender=gender, batch_size=batch_size, num_betas=num_betas,
                  use_pca=False, flat_hand_mean=True)
    
    return smplx_data, model

def generate_smplx_frames(smplx_data, smplx_model, frame_indices, batch_size=32, device='cpu', num_betas=16):
    """Generate SMPLX meshes for multiple frames in batches."""
    # Process in batches
    joints_all = []
    global_orient_all = []
    
    for i in tqdm(range(0, len(frame_indices), batch_size), desc="Generating SMPL-X frames"):
        batch_indices = frame_indices[i:i + batch_size]
        batch_size_actual = len(batch_indices)

        # Handle padding if needed
        if batch_size_actual < batch_size:
            last_frame_idx = batch_indices[-1]
            batch_indices_padded = batch_indices + [last_frame_idx] * (batch_size - batch_size_actual)
        else:
            batch_indices_padded = batch_indices

        batch_inputs = {
            'transl': torch.tensor(smplx_data['trans'][batch_indices_padded], dtype=torch.float32, device=device),
            'global_orient': torch.tensor(smplx_data['root_orient'][batch_indices_padded], dtype=torch.float32, device=device),
            'body_pose': torch.tensor(smplx_data['pose_body'][batch_indices_padded], dtype=torch.float32, device=device),
            'betas': torch.tensor(smplx_data['betas'][None, :num_betas], dtype=torch.float32, device=device),
            'jaw_pose': torch.tensor(smplx_data['pose_jaw'][batch_indices_padded], dtype=torch.float32, device=device),
            'leye_pose': torch.tensor(smplx_data['pose_eye'][batch_indices_padded, :3], dtype=torch.float32, device=device).unsqueeze(0),
            'reye_pose': torch.tensor(smplx_data['pose_eye'][batch_indices_padded, 3:], dtype=torch.float32, device=device).unsqueeze(0),
        }

        with torch.no_grad():
            output = smplx_model(**batch_inputs)
            
        # ---- Collect & unpad ----
        joints = output.joints[:batch_size_actual]              # (B, J, 3)
        global_orient = batch_inputs['global_orient'][:batch_size_actual]

        joints_all.append(joints.cpu().numpy())
        global_orient_all.append(global_orient.cpu().numpy())

    return {
        'joints': np.concatenate(joints_all, axis=0),            # (N, J, 3)
        'global_orient': np.concatenate(global_orient_all, axis=0)  # (N, 3)
    }

def process_file(smplx_file, output_dir, batch_size=128, num_betas=16):
    """Optimized processing of a single subject and take using batching."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
      
    # Load data
    smplx_data, smplx_model = load_data(smplx_file, batch_size=batch_size, num_betas=num_betas)
    
    # Move model to device
    smplx_model = smplx_model.to(device)
    
    # Setup output directories
    take_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(smplx_file))[0])
    os.makedirs(take_output_dir, exist_ok=True)
    
    # Determine frames to process
    total_frames = smplx_data['root_orient'].shape[0]
    frame_indices = list(range(0, total_frames))
    
    # Process all frames in batches
    print("Generating SMPL-X meshes...")
    smplx_outputs = generate_smplx_frames(
        smplx_data, smplx_model, frame_indices, batch_size, device=device, num_betas=num_betas
    )
    # Save the smplx output dict
    output_path = os.path.join(take_output_dir, f"{os.path.splitext(os.path.basename(smplx_file))[0]}_smplx_output.npz")
    np.savez_compressed(output_path, **smplx_outputs)
    print(f"Saved SMPL-X output to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process SMPLX data and extract CoM")
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to folder containing stageii .npz files (e.g., 35)")
    parser.add_argument('--output_dir', type=str, default='smplx_output',
                        help="Output directory")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size for processing frames")
    parser.add_argument('--num_betas', type=int, default=16, help="Number of shape coefficients to use")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # List all *_stageii.npz files in the folder
    npz_files = sorted(glob(os.path.join(args.input_folder, "*_stageii.npz")))

    print(f"Found {len(npz_files)} takes in {args.input_folder}")
    for npz_file in npz_files:
        print(f"Processing {os.path.basename(npz_file)}")
        process_file(
            smplx_file=npz_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_betas=args.num_betas
        )

                


            

if __name__ == "__main__":
    main()