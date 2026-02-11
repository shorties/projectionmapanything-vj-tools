"""Calibration data storage and retrieval for ProCam calibration."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from scope_depth.calibration.calibrator import (
    CameraIntrinsics,
    ProCamCalibration,
)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


@dataclass
class ProCamCalibrationData:
    """Serializable ProCam calibration data."""

    # Camera intrinsics
    camera_K: list
    camera_dist_coeffs: list
    camera_image_size: Tuple[int, int]
    camera_reprojection_error: float

    # Projector intrinsics
    projector_K: list
    projector_dist_coeffs: list
    projector_image_size: Tuple[int, int]

    # Stereo extrinsics
    R: list  # Rotation matrix (3x3)
    T: list  # Translation vector (3,)
    E: list  # Essential matrix (3x3)
    F: list  # Fundamental matrix (3x3)

    # Validation
    stereo_error: float
    is_valid: bool

    version: int = 2  # Version 2 = ProCam calibration

    @classmethod
    def from_calibration(cls, procam: ProCamCalibration) -> "ProCamCalibrationData":
        """Create serializable data from ProCamCalibration."""
        return cls(
            camera_K=procam.camera.K.tolist(),
            camera_dist_coeffs=procam.camera.dist_coeffs.tolist(),
            camera_image_size=procam.camera.image_size,
            camera_reprojection_error=procam.camera.reprojection_error,
            projector_K=procam.projector.K.tolist(),
            projector_dist_coeffs=procam.projector.dist_coeffs.tolist(),
            projector_image_size=procam.projector.image_size,
            R=procam.R.tolist(),
            T=procam.T.tolist(),
            E=procam.E.tolist(),
            F=procam.F.tolist(),
            stereo_error=procam.stereo_error,
            is_valid=procam.is_valid,
        )

    def to_calibration(self) -> ProCamCalibration:
        """Convert to ProCamCalibration object."""
        camera = CameraIntrinsics(
            K=np.array(self.camera_K, dtype=np.float32),
            dist_coeffs=np.array(self.camera_dist_coeffs, dtype=np.float32),
            image_size=self.camera_image_size,
            reprojection_error=self.camera_reprojection_error,
        )

        projector = CameraIntrinsics(
            K=np.array(self.projector_K, dtype=np.float32),
            dist_coeffs=np.array(self.projector_dist_coeffs, dtype=np.float32),
            image_size=self.projector_image_size,
            reprojection_error=0.0,
        )

        return ProCamCalibration(
            camera=camera,
            projector=projector,
            R=np.array(self.R, dtype=np.float32),
            T=np.array(self.T, dtype=np.float32),
            E=np.array(self.E, dtype=np.float32),
            F=np.array(self.F, dtype=np.float32),
            stereo_error=self.stereo_error,
            is_valid=self.is_valid,
        )


class CalibrationStorage:
    """Handles saving/loading ProCam calibration data."""

    DEFAULT_FILENAME = "procam_calibration.json"
    DEPTH_MAP_FILENAME = "calibration_depth_map.png"

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize storage.

        Args:
            storage_dir: Directory for calibration files. If None, uses current dir.
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path.cwd()

        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_calibration(
        self,
        procam: ProCamCalibration,
        filename: Optional[str] = None,
    ) -> str:
        """Save ProCam calibration to JSON file.

        Args:
            procam: ProCamCalibration object
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = self.DEFAULT_FILENAME

        filepath = self.storage_dir / filename

        data = ProCamCalibrationData.from_calibration(procam)

        with open(filepath, "w") as f:
            json.dump(asdict(data), f, indent=2, cls=NumpyEncoder)

        return str(filepath)

    def load_calibration(
        self,
        filename: Optional[str] = None,
    ) -> Optional[ProCamCalibration]:
        """Load ProCam calibration from JSON file.

        Args:
            filename: Optional custom filename

        Returns:
            ProCamCalibration or None if not found/invalid
        """
        if filename is None:
            filename = self.DEFAULT_FILENAME

        filepath = self.storage_dir / filename

        if not filepath.exists():
            return None

        try:
            with open(filepath) as f:
                data_dict = json.load(f)

            # Handle version migration if needed
            version = data_dict.get("version", 1)
            if version < 2:
                # Old format - return None (requires recalibration)
                return None

            data = ProCamCalibrationData(**data_dict)
            return data.to_calibration()
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Error loading calibration: {e}")
            return None

    def has_calibration(self, filename: Optional[str] = None) -> bool:
        """Check if calibration file exists.

        Args:
            filename: Optional custom filename

        Returns:
            True if calibration exists
        """
        if filename is None:
            filename = self.DEFAULT_FILENAME
        return (self.storage_dir / filename).exists()

    def delete_calibration(self, filename: Optional[str] = None) -> bool:
        """Delete calibration file.

        Args:
            filename: Optional custom filename

        Returns:
            True if file was deleted
        """
        if filename is None:
            filename = self.DEFAULT_FILENAME

        filepath = self.storage_dir / filename
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def export_depth_map(
        self,
        projector_size: Tuple[int, int],
        output_path: Optional[str] = None,
    ) -> str:
        """Export a default/identity depth map for ControlNet.

        Args:
            projector_size: (width, height) of projector
            output_path: Optional custom output path

        Returns:
            Path to exported depth map
        """
        if output_path is None:
            output_path = self.storage_dir / self.DEPTH_MAP_FILENAME

        width, height = projector_size
        depth_map = np.ones((height, width), dtype=np.uint8) * 128

        cv2.imwrite(str(output_path), depth_map)

        return str(output_path)

    def save_warped_depth_preview(
        self,
        depth_tensor: torch.Tensor,
        output_path: Optional[str] = None,
    ) -> str:
        """Save a preview of the warped depth for verification.

        Args:
            depth_tensor: Warped depth tensor (1, H, W, 3) or (H, W, 3)
            output_path: Optional custom output path

        Returns:
            Path to saved image
        """
        if output_path is None:
            output_path = self.storage_dir / "depth_preview.png"

        if len(depth_tensor.shape) == 4:
            depth = depth_tensor[0]
        else:
            depth = depth_tensor

        depth_np = depth.cpu().numpy()
        depth_gray = (depth_np.mean(axis=2) * 255).astype(np.uint8)

        cv2.imwrite(str(output_path), depth_gray)

        return str(output_path)

    def get_calibration_info(self) -> dict:
        """Get information about stored calibration.

        Returns:
            Dict with calibration status and metadata
        """
        filepath = self.storage_dir / self.DEFAULT_FILENAME

        if not filepath.exists():
            return {
                "has_calibration": False,
                "file_path": str(filepath),
                "projector_size": None,
                "error": None,
            }

        procam = self.load_calibration()
        if procam is None:
            return {
                "has_calibration": False,
                "file_path": str(filepath),
                "projector_size": None,
                "error": "Invalid or outdated calibration file",
            }

        return {
            "has_calibration": True,
            "file_path": str(filepath),
            "projector_size": procam.projector.image_size,
            "camera_size": procam.camera.image_size,
            "camera_error": procam.camera.reprojection_error,
            "stereo_error": procam.stereo_error,
            "is_valid": procam.is_valid,
        }
