from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace


@dataclass(slots=True)
class FaceActionResult:
    face_found: bool

    head_direction: str = "unknown"
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    left_eye_closed: bool = False
    right_eye_closed: bool = False
    both_eyes_closed: bool = False

    smiling: bool = False

    left_blink_score: float = 0.0
    right_blink_score: float = 0.0
    smile_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FaceActionAnalyzer:
    """
    Face-action analysis using MediaPipe Face Landmarker.

    Input frames are expected in OpenCV BGR format.

    Left/right eye names refer to the PERSON'S anatomical left/right.
    If your preview is mirrored, the displayed sides will look reversed.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        blink_threshold: float = 0.55,
        smile_threshold: float = 0.45,
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 15.0,
        invert_yaw: bool = False,
    ) -> None:
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found: {model_path}"
            )

        self.blink_threshold = float(blink_threshold)
        self.smile_threshold = float(smile_threshold)
        self.yaw_threshold = float(yaw_threshold)
        self.pitch_threshold = float(pitch_threshold)
        self.invert_yaw = bool(invert_yaw)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

        self._landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceActionAnalyzer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _to_mediapipe_image(frame_bgr: np.ndarray) -> mp.Image:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

    @staticmethod
    def _blendshape_scores(result) -> dict[str, float]:
        if not result.face_blendshapes:
            return {}

        return {
            category.category_name: float(category.score)
            for category in result.face_blendshapes[0]
        }

    @staticmethod
    def _head_pose(result) -> tuple[float, float, float]:
        """
        Returns (pitch, yaw, roll) in degrees.
        """
        if not result.facial_transformation_matrixes:
            return 0.0, 0.0, 0.0

        matrix = np.asarray(
            result.facial_transformation_matrixes[0],
            dtype=np.float64,
        )

        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            return 0.0, 0.0, 0.0

        rotation = matrix[:3, :3]

        angles = cv2.RQDecomp3x3(rotation)[0]

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return pitch, yaw, roll

    def analyze(self, frame_bgr: np.ndarray) -> FaceActionResult:
        image = self._to_mediapipe_image(frame_bgr)
        result = self._landmarker.detect(image)

        if not result.face_landmarks:
            return FaceActionResult(face_found=False)

        scores = self._blendshape_scores(result)

        left_blink = scores.get("eyeBlinkLeft", 0.0)
        right_blink = scores.get("eyeBlinkRight", 0.0)

        smile_left = scores.get("mouthSmileLeft", 0.0)
        smile_right = scores.get("mouthSmileRight", 0.0)
        smile_score = (smile_left + smile_right) / 2.0

        pitch, yaw, roll = self._head_pose(result)

        if self.invert_yaw:
            yaw = -yaw

        if yaw <= -self.yaw_threshold:
            head_direction = "left"
        elif yaw >= self.yaw_threshold:
            head_direction = "right"
        elif pitch <= -self.pitch_threshold:
            head_direction = "up"
        elif pitch >= self.pitch_threshold:
            head_direction = "down"
        else:
            head_direction = "center"

        left_eye_closed = left_blink >= self.blink_threshold
        right_eye_closed = right_blink >= self.blink_threshold

        return FaceActionResult(
            face_found=True,
            head_direction=head_direction,
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            left_eye_closed=left_eye_closed,
            right_eye_closed=right_eye_closed,
            both_eyes_closed=(
                left_eye_closed and right_eye_closed
            ),
            smiling=smile_score >= self.smile_threshold,
            left_blink_score=round(left_blink, 4),
            right_blink_score=round(right_blink, 4),
            smile_score=round(smile_score, 4),
        )

    # ---------------------------------------------------------
    # Convenience action checks
    # ---------------------------------------------------------

    def is_head_left(self, frame_bgr: np.ndarray) -> bool:
        result = self.analyze(frame_bgr)
        return result.face_found and result.head_direction == "left"

    def is_head_right(self, frame_bgr: np.ndarray) -> bool:
        result = self.analyze(frame_bgr)
        return result.face_found and result.head_direction == "right"

    def is_left_eye_closed(self, frame_bgr: np.ndarray) -> bool:
        result = self.analyze(frame_bgr)
        return result.face_found and result.left_eye_closed

    def is_right_eye_closed(self, frame_bgr: np.ndarray) -> bool:
        result = self.analyze(frame_bgr)
        return result.face_found and result.right_eye_closed

    def are_both_eyes_closed(self, frame_bgr: np.ndarray) -> bool:
        result = self.analyze(frame_bgr)
        return result.face_found and result.both_eyes_closed

    def is_smiling(self, frame_bgr: np.ndarray) -> bool:
        result = self.analyze(frame_bgr)
        return result.face_found and result.smiling

    # ---------------------------------------------------------
    # Face recognition / verification
    # ---------------------------------------------------------

    @staticmethod
    def verify_face(
        reference_image: str | Path | np.ndarray,
        frame_bgr: np.ndarray,
        *,
        model_name: str = "Facenet512",
        detector_backend: str = "retinaface",
        distance_metric: str = "cosine",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Compare a known/reference face against the current camera frame.

        Do NOT call this on every video frame unless you really need to.
        It is much heavier than MediaPipe action analysis.
        """
        result = DeepFace.verify(
            img1_path=(
                str(reference_image)
                if isinstance(reference_image, Path)
                else reference_image
            ),
            img2_path=frame_bgr,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=True,
            align=True,
            threshold=threshold,
        )

        return {
            "verified": bool(result.get("verified", False)),
            "distance": float(result.get("distance", 0.0)),
            "threshold": float(result.get("threshold", 0.0)),
            "model": model_name,
            "detector": detector_backend,
            "metric": distance_metric,
        }

    def check_action(
        self,
        frame_bgr: np.ndarray,
        action: str,
    ) -> dict[str, Any]:
        """
        Supported actions:
          - turn_left
          - turn_right
          - look_center
          - close_left_eye
          - close_right_eye
          - close_both_eyes
          - smile
        """
        result = self.analyze(frame_bgr)

        if not result.face_found:
            return {
                "ok": False,
                "action": action,
                "reason": "no_face",
                "analysis": result.to_dict(),
            }

        checks = {
            "turn_left": result.head_direction == "left",
            "turn_right": result.head_direction == "right",
            "look_center": result.head_direction == "center",
            "close_left_eye": result.left_eye_closed,
            "close_right_eye": result.right_eye_closed,
            "close_both_eyes": result.both_eyes_closed,
            "smile": result.smiling,
        }

        if action not in checks:
            raise ValueError(f"Unsupported face action: {action}")

        return {
            "ok": bool(checks[action]),
            "action": action,
            "reason": "ok" if checks[action] else "action_not_detected",
            "analysis": result.to_dict(),
        }
