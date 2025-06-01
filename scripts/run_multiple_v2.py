#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa
import torch, gc

def parse_args():
	parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")

	parser.add_argument("files", nargs="*", help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.")
	parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS) # deprecated
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", "--snapshot", default="", help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .ingp/.msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, but helps with high PSNR on synthetic scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1), metavar=("START_FRAME", "END_FRAME"), help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).")

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")
	parser.add_argument("--marching_cubes_density_thresh", default=2.5, type=float, help="Sets the density threshold for marching cubes.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")
	parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")

	parser.add_argument("--model_id_start", default=1, help=".")
	parser.add_argument("--model_id_max", default=47, help=".")
	parser.add_argument("--model_id_step", default=1, help=".")
	parser.add_argument("--model_subpath", default="3000.ingp", help=".")


	return parser.parse_args()

def get_scene(scene):
	for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene in scenes:
			return scenes[scene]
	return None

def calculate_model_id(frame_index, cycle_per, model_id_start, model_id_max, model_id_step):
	"""Calculate which model_id to use for a given frame"""
	if cycle_per % 1 == 0:  # Every frame (keeping original logic)
		model_id = model_id_start + (frame_index * model_id_step)
		if model_id > model_id_max:
			# Cycle back to start
			cycle_offset = (frame_index * model_id_step) % (model_id_max - model_id_start + 1)
			model_id = model_id_start + cycle_offset
	else:
		model_id = model_id_start
	return model_id

def init_single_testbed(args):
	"""Initialize testbed once, to be reused for all frames"""
	testbed = ngp.Testbed()
	testbed.root_dir = ROOT_DIR

	if testbed.mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	testbed.shall_train = args.train if args.gui else True
	testbed.nerf.render_with_lens_distortion = True

	# Load camera path
	testbed.load_camera_path(args.video_camera_path)
	testbed.camera_smoothing = args.video_camera_smoothing
	
	return testbed

if __name__ == "__main__":
	args = parse_args()
	if args.vr: # VR implies having the GUI running at the moment
		args.gui = True

	if args.mode:
		print("Warning: the '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.")

	if "tmp" in os.listdir():
		shutil.rmtree("tmp")
	os.makedirs("tmp")

	if args.video_camera_path:
		resolution = [args.width or 1920, args.height or 1080]
		n_frames = args.video_n_seconds * args.video_fps
		save_frames = "%" in args.video_output
		start_frame, end_frame = args.video_render_range

		# ===== MAIN CHANGE: Single testbed for all frames =====
		print("Initializing single testbed for all frames...")
		testbed = init_single_testbed(args)
		
		current_model_id = None
		cycle_per = 0
		model_id_start = int(args.model_id_start)
		model_id_max = int(args.model_id_max)
		model_id_step = int(args.model_id_step)
		
		print(f"Rendering {n_frames} frames with model switching...")
		
		for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
			# Calculate which model should be used for this frame
			new_model_id = calculate_model_id(i, cycle_per, model_id_start, model_id_max, model_id_step)
			
			# Only load new snapshot if model changed
			if new_model_id != current_model_id:
				snapshot_path = "/content/drive/MyDrive/kek_v/frame_{}/{}".format(new_model_id, args.model_subpath)
				print(f"Loading model {new_model_id} for frame {i}")
				
				# Check if snapshot exists
				if not os.path.exists(snapshot_path):
					print(f"Warning: Snapshot {snapshot_path} not found, skipping...")
					continue
					
				# Load new model (preserves internal state like training_step, etc.)
				scene_info = get_scene(snapshot_path)
				if scene_info is not None:
					snapshot_path = default_snapshot_filename(scene_info)
				testbed.load_snapshot(snapshot_path)
				
				current_model_id = new_model_id
			
			# Render frame (testbed keeps its "warmed up" state)
			frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, 
								 float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, 
								 shutter_fraction=0.5)

			assert "tmp" in os.listdir()

			if save_frames:
				write_image(args.video_output % i, np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)
			else:
				write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

			# No deletion of testbed here - it lives for the entire process!

		# Clean up memory only at the very end
		print("Cleaning up...")
		del testbed
		torch.cuda.empty_cache()
		gc.collect()

		if not save_frames:
			print("Creating video with ffmpeg...")
			os.system(f"ffmpeg -y -framerate {args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {args.video_output}")

		print(f"Video generation complete: {args.video_output}")