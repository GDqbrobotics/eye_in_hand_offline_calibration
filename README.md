# Offline eye-in-hand calibration

This script calibrates camera intrinsics and the transform from the robot's
`tool0` frame to the camera optical frame using saved JPG images and matching
robot-pose JSON files. It runs offline and does not connect to the camera or
robot.

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

The board settings in the script are a 25-column by 18-row ChArUco board using
the 5x5, 250-marker ArUco dictionary, 7.5 mm squares, and 5.25 mm markers. The
robot-frame-to-optical-frame axis conversion also assumes the configured
`tool0` axes: +X right in the image and +Y out through the lens.

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

The script refuses to save calibration when intrinsic RMS exceeds 5 px. A
fixed-board consistency RMS above 10 mm is reported as a warning; in that case,
do not treat the hand-eye transform as validated.
