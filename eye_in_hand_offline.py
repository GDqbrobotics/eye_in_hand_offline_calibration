import argparse
import json
import sys
from pathlib import Path

import cv2
from cv2 import aruco
import numpy as np


def _vector3(value, name):
    if isinstance(value, dict):
        try:
            vector = [value[axis] for axis in ("x", "y", "z")]
        except KeyError as exc:
            raise ValueError(f"{name} must contain x, y, and z") from exc
    else:
        vector = value
    result = np.asarray(vector, dtype=np.float64).reshape(-1)
    if result.size != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be three finite numbers")
    return result


def _quaternion_xyzw(value):
    if isinstance(value, dict):
        if "quaternion" in value:
            value = value["quaternion"]
        elif "xyzw" in value:
            value = value["xyzw"]
        else:
            try:
                value = [value[axis] for axis in ("x", "y", "z", "w")]
            except KeyError as exc:
                raise ValueError(
                    "orientation must contain quaternion x, y, z, and w"
                ) from exc

    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.size != 4 or not np.all(np.isfinite(quaternion)):
        raise ValueError("orientation quaternion must be four finite numbers")
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("orientation quaternion must not be zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_to_quaternion_xyzw(rotation):
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")

    trace = np.trace(matrix)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                scale / 4,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.array(
                [
                    scale / 4,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    scale / 4,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    scale / 4,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def _find_pose_pair(value):
    if isinstance(value, dict):
        lowered_keys = {str(key).lower(): key for key in value}
        position_key = lowered_keys.get("position")
        orientation_key = lowered_keys.get("orientation")
        if position_key is not None:
            position = value[position_key]
            if orientation_key is not None:
                return position, value[orientation_key]
            combined_pose = np.asarray(position, dtype=np.float64).reshape(-1)
            if combined_pose.size == 7:
                return combined_pose[:3], combined_pose[3:]

        for nested in value.values():
            try:
                return _find_pose_pair(nested)
            except ValueError:
                continue
    raise ValueError("could not find an end-effector pose with position and orientation")


def _find_pose_object(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if "end" in normalized and "effector" in normalized and "pose" in normalized:
                return _find_pose_pair(nested)
        for nested in value.values():
            try:
                return _find_pose_object(nested)
            except ValueError:
                continue
    return _find_pose_pair(value)


def load_base_to_gripper_pose(json_path):
    with json_path.open("r", encoding="utf-8") as pose_file:
        data = json.load(pose_file)

    position_value, orientation_value = _find_pose_object(data)
    position = _vector3(position_value, "position")
    rotation = _quaternion_xyzw(orientation_value)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    try:
        board_config = config["board"]
        frame_config = config["frames"]
        calibration_config = config["calibration"]
        debug_config = config["debug"]
        config["paths"]
        board_size = (
            int(board_config["squares_x"]),
            int(board_config["squares_y"]),
        )
        square_length = float(board_config["square_length_m"])
        marker_length = float(board_config["marker_length_m"])
        dictionary_name = board_config["dictionary"]
        tool0_to_optical = np.asarray(
            frame_config["tool0_to_optical"], dtype=np.float64
        )
        parent_frame = frame_config["parent"]
        child_frame = frame_config["child"]
        paths_config = config["paths"]
        image_path = paths_config["images"]
        poses_path = paths_config["poses"]
        output_path = paths_config["output"]
        tf2_output_path = paths_config.get("tf2_output")
        debug_dir = paths_config.get("debug_dir")
        minimum_valid_images = int(calibration_config["minimum_valid_images"])
        minimum_charuco_corners = int(
            calibration_config["minimum_charuco_corners"]
        )
        hand_eye_method_name = calibration_config["hand_eye_method"]
        max_intrinsic_rms = float(calibration_config["max_intrinsic_rms_px"])
        max_board_consistency = float(
            calibration_config["max_board_translation_consistency_rms_m"]
        )
        pixels_per_square = int(debug_config["pixels_per_square"])
        overlay_image_count = int(debug_config["overlay_image_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid calibration config {config_path}: {exc}") from exc

    if min(board_size) < 2:
        raise ValueError("board.squares_x and board.squares_y must be at least 2")
    if square_length <= 0 or marker_length <= 0 or marker_length >= square_length:
        raise ValueError(
            "Board lengths must be positive and marker_length_m must be "
            "smaller than square_length_m"
        )
    if not isinstance(dictionary_name, str) or not hasattr(aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name!r}")
    if not dictionary_name.startswith("DICT_"):
        raise ValueError("board.dictionary must be an OpenCV DICT_* constant name")
    if tool0_to_optical.shape != (4, 4) or not np.all(
        np.isfinite(tool0_to_optical)
    ):
        raise ValueError("frames.tool0_to_optical must be a finite 4x4 matrix")
    tool_to_optical_rotation = tool0_to_optical[:3, :3]
    if not np.allclose(
        tool_to_optical_rotation.T @ tool_to_optical_rotation,
        np.eye(3),
        atol=1e-6,
    ) or not np.isclose(np.linalg.det(tool_to_optical_rotation), 1.0, atol=1e-6):
        raise ValueError(
            "frames.tool0_to_optical rotation must be a right-handed "
            "orthonormal rotation matrix"
        )
    if not np.allclose(tool0_to_optical[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(
            "frames.tool0_to_optical must have homogeneous bottom row [0, 0, 0, 1]"
        )
    if not isinstance(parent_frame, str) or not parent_frame:
        raise ValueError("frames.parent must be a non-empty string")
    if not isinstance(child_frame, str) or not child_frame:
        raise ValueError("frames.child must be a non-empty string")
    if minimum_valid_images < 3:
        raise ValueError("calibration.minimum_valid_images must be at least 3")
    if minimum_charuco_corners < 4:
        raise ValueError("calibration.minimum_charuco_corners must be at least 4")
    if not isinstance(hand_eye_method_name, str) or not hand_eye_method_name.startswith(
        "CALIB_HAND_EYE_"
    ):
        raise ValueError(
            "calibration.hand_eye_method must be an OpenCV CALIB_HAND_EYE_* name"
        )
    if not hasattr(cv2, hand_eye_method_name):
        raise ValueError(f"Unknown OpenCV hand-eye method: {hand_eye_method_name}")
    if max_intrinsic_rms <= 0 or max_board_consistency <= 0:
        raise ValueError("Calibration error thresholds must be positive")
    if pixels_per_square < 1 or overlay_image_count < 0:
        raise ValueError("Debug image dimensions/count are invalid")
    for key, value in (
        ("images", image_path),
        ("poses", poses_path),
        ("output", output_path),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"paths.{key} must be a non-empty path string")
    for key, value in (("tf2_output", tf2_output_path), ("debug_dir", debug_dir)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"paths.{key} must be null or a non-empty path string")

    config["_resolved"] = {
        "board_size": board_size,
        "square_length_m": square_length,
        "marker_length_m": marker_length,
        "dictionary": getattr(aruco, dictionary_name),
        "tool0_to_optical": tool0_to_optical,
        "parent_frame": parent_frame,
        "child_frame": child_frame,
        "minimum_valid_images": minimum_valid_images,
        "minimum_charuco_corners": minimum_charuco_corners,
        "hand_eye_method": getattr(cv2, hand_eye_method_name),
        "max_intrinsic_rms_px": max_intrinsic_rms,
        "max_board_translation_consistency_rms_m": max_board_consistency,
        "pixels_per_square": pixels_per_square,
        "overlay_image_count": overlay_image_count,
        "config_dir": config_path.resolve().parent,
    }
    return config


def _configured_path(value, config):
    path = Path(value)
    if not path.is_absolute():
        path = config["_resolved"]["config_dir"] / path
    return path


def _make_charuco_board(config):
    resolved = config["_resolved"]
    dictionary = aruco.getPredefinedDictionary(resolved["dictionary"])
    board = aruco.CharucoBoard_create(
        resolved["board_size"][0],
        resolved["board_size"][1],
        resolved["square_length_m"],
        resolved["marker_length_m"],
        dictionary,
    )
    return dictionary, board


def _write_debug_artifacts(samples, images_dir, board, debug_dir, config):
    debug_dir.mkdir(parents=True, exist_ok=True)

    resolved = config["_resolved"]
    board_size = resolved["board_size"]
    pixels_per_square = resolved["pixels_per_square"]
    board_image = aruco.drawPlanarBoard(
        board,
        (board_size[0] * pixels_per_square, board_size[1] * pixels_per_square),
        0,
        1,
    )
    board_path = debug_dir / f"charuco_board_{board_size[0]}x{board_size[1]}.png"
    if not cv2.imwrite(str(board_path), board_image):
        raise OSError(f"Could not write generated board image: {board_path}")

    debug_samples = sorted(
        samples, key=lambda sample: len(sample["ids"]), reverse=True
    )[: resolved["overlay_image_count"]]
    written_images = []
    for sample in debug_samples:
        image_path = images_dir / sample["name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"Could not read image for debug overlay: {image_path}")

        aruco.drawDetectedMarkers(
            image, sample["marker_corners"], sample["marker_ids"]
        )
        aruco.drawDetectedCornersCharuco(
            image, sample["corners"], sample["ids"], (0, 255, 0)
        )
        cv2.putText(
            image,
            f"{sample['name']}: {len(sample['marker_ids'])} markers, "
            f"{len(sample['ids'])} ChArUco corners",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        output_image = debug_dir / f"detected_{sample['name']}"
        if not cv2.imwrite(str(output_image), image):
            raise OSError(f"Could not write debug overlay image: {output_image}")
        written_images.append(output_image)

    print(f"Saved generated ChArUco board to {board_path}")
    for output_image in written_images:
        print(f"Saved detection overlay to {output_image}")


def calibrate_from_directories(
    images_dir,
    poses_dir,
    output_path,
    tf2_output_path=None,
    parent_frame=None,
    child_frame=None,
    debug_dir=None,
    config=None,
):
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not poses_dir.is_dir():
        raise FileNotFoundError(f"Poses directory does not exist: {poses_dir}")

    image_paths = sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"),
        key=lambda path: path.name,
    )
    if not image_paths:
        raise ValueError(f"No .jpg images found in {images_dir}")

    if config is None:
        config = load_config(Path(__file__).resolve().with_name("calibration_config.json"))
    resolved = config["_resolved"]
    parent_frame = parent_frame or resolved["parent_frame"]
    child_frame = child_frame or resolved["child_frame"]
    dictionary, board = _make_charuco_board(config)
    samples = []
    image_size = None

    for image_path in image_paths:
        pose_path = poses_dir / f"{image_path.stem}.json"
        if not pose_path.is_file():
            print(f"Skipping {image_path.name}: matching pose file not found ({pose_path})", file=sys.stderr)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping {image_path.name}: image could not be read", file=sys.stderr)
            continue

        try:
            base_to_gripper = load_base_to_gripper_pose(pose_path)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"Skipping {image_path.name}: invalid pose JSON: {exc}", file=sys.stderr)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(
                f"Image {image_path.name} has size {current_size}; expected {image_size}"
            )

        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)
        if marker_ids is None:
            print(f"Skipping {image_path.name}: no board markers detected", file=sys.stderr)
            continue

        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < resolved["minimum_charuco_corners"]
        ):
            print(
                f"Skipping {image_path.name}: fewer than "
                f"{resolved['minimum_charuco_corners']} ChArUco corners detected",
                file=sys.stderr,
            )
            continue

        samples.append(
            {
                "name": image_path.name,
                "marker_corners": marker_corners,
                "marker_ids": marker_ids,
                "corners": charuco_corners,
                "ids": charuco_ids,
                "base_to_gripper": base_to_gripper,
            }
        )

    if debug_dir is None:
        debug_dir = output_path.with_name(f"{output_path.stem}_debug")
    if samples:
        _write_debug_artifacts(samples, images_dir, board, debug_dir, config)

    if len(samples) < resolved["minimum_valid_images"]:
        raise ValueError(
            f"Need at least {resolved['minimum_valid_images']} matched images "
            "with valid poses and ChArUco detections; "
            f"found {len(samples)}"
        )

    rms, camera_matrix, distortion_coefficients, _, _ = aruco.calibrateCameraCharuco(
        [sample["corners"] for sample in samples],
        [sample["ids"] for sample in samples],
        board,
        image_size,
        None,
        None,
    )
    if not np.isfinite(rms) or rms > resolved["max_intrinsic_rms_px"]:
        raise ValueError(
            f"Intrinsic calibration RMS is {rms:.3f} px; refusing to save an "
            "unreliable calibration "
            f"(expected at most {resolved['max_intrinsic_rms_px']} px)"
        )

    base_to_optical_rotations = []
    base_to_optical_translations = []
    target_to_camera_rotations = []
    target_to_camera_translations = []
    used_base_to_tool0 = []

    for sample in samples:
        pose_found, rvec, tvec = aruco.estimatePoseCharucoBoard(
            sample["corners"],
            sample["ids"],
            board,
            camera_matrix,
            distortion_coefficients,
            None,
            None,
        )
        if not pose_found:
            print(f"Skipping {sample['name']}: could not estimate board pose", file=sys.stderr)
            continue

        # Convert the JSON tool0 pose to an optical-axis frame at the same origin.
        base_to_gripper = sample["base_to_gripper"]
        base_to_optical = base_to_gripper @ resolved["tool0_to_optical"]
        base_to_optical_rotations.append(base_to_optical[:3, :3])
        base_to_optical_translations.append(
            base_to_optical[:3, 3].reshape(3, 1)
        )
        target_to_camera_rotations.append(cv2.Rodrigues(rvec)[0])
        target_to_camera_translations.append(np.asarray(tvec).reshape(3, 1))
        used_base_to_tool0.append(base_to_gripper)

    if len(base_to_optical_rotations) < 3:
        raise ValueError(
            "Need at least 3 images with successful board pose estimates for hand-eye calibration"
        )

    optical_to_camera_rotation, optical_to_camera_translation = cv2.calibrateHandEye(
        base_to_optical_rotations,
        base_to_optical_translations,
        target_to_camera_rotations,
        target_to_camera_translations,
        method=resolved["hand_eye_method"],
    )

    optical_to_camera = np.eye(4, dtype=np.float64)
    optical_to_camera[:3, :3] = optical_to_camera_rotation
    optical_to_camera[:3, 3] = optical_to_camera_translation.reshape(3)

    tool0_to_camera = resolved["tool0_to_optical"] @ optical_to_camera

    board_positions_in_base = []
    for base_to_tool0, target_rotation, target_translation in zip(
        used_base_to_tool0,
        target_to_camera_rotations,
        target_to_camera_translations,
    ):
        camera_to_board = np.eye(4, dtype=np.float64)
        camera_to_board[:3, :3] = target_rotation
        camera_to_board[:3, 3] = target_translation.reshape(3)
        board_in_base = base_to_tool0 @ tool0_to_camera @ camera_to_board
        board_positions_in_base.append(board_in_base[:3, 3])

    board_positions_in_base = np.asarray(board_positions_in_base)
    board_position_mean = board_positions_in_base.mean(axis=0)
    board_translation_consistency_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (board_positions_in_base - board_position_mean) ** 2,
                    axis=1,
                )
            )
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"Could not open calibration output for writing: {output_path}")
    try:
        storage.write("camera_matrix", camera_matrix)
        storage.write("distortion_coefficients", distortion_coefficients)
        storage.write("tool0_to_camera", tool0_to_camera)
        storage.write("camera_to_gripper", tool0_to_camera)
        storage.write("intrinsic_rms_reprojection_error", float(rms))
        storage.write("calibration_image_count", len(base_to_optical_rotations))
        storage.write(
            "board_translation_consistency_rms_m",
            board_translation_consistency_rms,
        )
    finally:
        storage.release()

    quaternion = _rotation_to_quaternion_xyzw(tool0_to_camera[:3, :3])
    translation = tool0_to_camera[:3, 3]
    tf2_command = (
        "ros2 run tf2_ros static_transform_publisher "
        f"--x {translation[0]:.9g} --y {translation[1]:.9g} "
        f"--z {translation[2]:.9g} "
        f"--qx {quaternion[0]:.9g} --qy {quaternion[1]:.9g} "
        f"--qz {quaternion[2]:.9g} --qw {quaternion[3]:.9g} "
        f"--frame-id {parent_frame} --child-frame-id {child_frame}"
    )
    if tf2_output_path is None:
        tf2_output_path = output_path.with_name(f"{output_path.stem}_tf2.txt")
    tf2_output_path.parent.mkdir(parents=True, exist_ok=True)
    tf2_output_path.write_text(tf2_command + "\n", encoding="utf-8")

    print(f"Calibrated using {len(base_to_optical_rotations)} image/pose pairs")
    print(f"Intrinsic RMS reprojection error: {rms:.6f} px")
    print(
        "Fixed-board translation consistency RMS: "
        f"{board_translation_consistency_rms * 1000:.2f} mm"
    )
    if board_translation_consistency_rms > resolved[
        "max_board_translation_consistency_rms_m"
    ]:
        print(
            "WARNING: board consistency exceeds configured threshold "
            f"({resolved['max_board_translation_consistency_rms_m'] * 1000:.1f} mm); "
            "do not treat the hand-eye transform as validated."
        )
    print(f"Saved camera intrinsics and tool0-to-camera transform to {output_path}")
    print(f"Saved ROS 2 TF command to {tf2_output_path}:")
    print(tf2_command)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Offline ChArUco camera and eye-in-hand calibration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "calibration_config.json",
        help="JSON calibration configuration file",
    )
    parser.add_argument("--images", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tf2-output",
        type=Path,
        help="Path for the ROS 2 static_transform_publisher command "
        "(default: <output-stem>_tf2.txt)",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Directory for generated board and detected-image overlays "
        "(default: <output-stem>_debug)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    images_dir = args.images or _configured_path(paths["images"], config)
    poses_dir = args.poses or _configured_path(paths["poses"], config)
    output_path = args.output or _configured_path(paths["output"], config)
    tf2_output_path = args.tf2_output
    if tf2_output_path is None and paths.get("tf2_output"):
        tf2_output_path = _configured_path(paths["tf2_output"], config)
    debug_dir = args.debug_dir
    if debug_dir is None and paths.get("debug_dir"):
        debug_dir = _configured_path(paths["debug_dir"], config)
    calibrate_from_directories(
        images_dir,
        poses_dir,
        output_path,
        tf2_output_path=tf2_output_path,
        parent_frame=config["frames"]["parent"],
        child_frame=config["frames"]["child"],
        debug_dir=debug_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
