# Offline eye-in-hand calibration

This script calibrates camera intrinsics and the transform from the robot's
`tool0` frame to the camera optical frame using saved JPG images and matching
robot-pose JSON files. It runs offline and does not connect to the camera or
robot. Calibration parameters are stored in `calibration_config.json`.

## Requirements

- Python 3
- NumPy
- OpenCV with the `aruco` module (for example, `opencv-contrib-python`)

## Input files

By default, the script reads JPGs directly from `images/` and matching JSON
files directly from `poses/`. File stems must match:

```text
images/object_calibration1_wp_1.jpg
poses/object_calibration1_wp_1.json
```

The JSON `position` field can be a seven-value array:

```json
"position": [x, y, z, qx, qy, qz, qw]
```

Translation is interpreted in metres and the quaternion in xyzw order. The
robot pose is expected to describe the end-effector in the base frame.

The default config describes a 25-column by 18-row ChArUco board using the
5x5, 250-marker ArUco dictionary, 7.5 mm squares, and 5.25 mm markers. It also
defines the tool-to-optical rotation, TF frame names, calibration thresholds,
debug output settings, and default paths. Edit this file to match your board
and robot/camera frames.

`frames.tool0_to_optical` is a 4x4 homogeneous transform from the JSON tool
frame to the camera optical-axis frame, including rotation and translation.
It follows the parent-to-child transform convention: its rotation columns
describe the optical-frame axes expressed in tool0 coordinates. The default
aligns optical +X with tool0 +X, optical +Y with tool0 -Z, and optical +Z with
tool0 +Y, with coincident origins. Set its translation column to the measured
optical-origin position expressed in tool0 if the frame origins differ.

The dictionary uses an OpenCV ArUco name such as `DICT_5X5_250`. The board
dimensions, square/marker lengths, TF frame names, hand-eye method, detection
and pose-count thresholds, error thresholds, debug image settings, and default
input/output paths can also be edited in the JSON file. The transform must be
a valid 4x4 rigid transform.

## Run

From the project directory:

```bash
python3 eye_in_hand_offline.py
```

Defaults are `images/`, `poses/`, and `calibration.yaml`. To use another data
set or choose output paths:

```bash
python3 eye_in_hand_offline.py \
  --images images/example_set \
  --poses poses/example_set \
  --output results/calibration.yaml \
  --tf2-output results/camera_tf2.txt \
  --debug-dir results/debug
```

Paths can be relative to the current working directory or absolute.
Use `--config path/to/config.json` to load an alternate configuration file.
Relative paths inside a config file are resolved relative to that config file.
CLI path arguments override paths configured in the JSON file.

## Outputs

- `calibration.yaml`: camera matrix, distortion coefficients, the
  `tool0_to_camera` transform, intrinsic reprojection RMS, image count, and
  fixed-board translation consistency RMS.
- `calibration_tf2.txt`: a ROS 2 `static_transform_publisher` command with
  `tool0` as the parent and `camera_link` as the child.
- `calibration_debug/`: generated board image and overlays for the two images
  with the most detected ChArUco corners.

The board PNG is an image for visual comparison. For a correctly sized print,
print it without scaling at approximately 271 DPI; its physical size is
187.5 x 135 mm.

The script refuses to save calibration when intrinsic RMS exceeds the
configured `max_intrinsic_rms_px`. A fixed-board consistency RMS above
`max_board_translation_consistency_rms_m` is reported as a warning; in that
case, do not treat the hand-eye transform as validated.
