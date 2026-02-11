"""Depth Webcam Pipeline with ProCam calibration for projection mapping."""

from typing import TYPE_CHECKING, Any, List

import numpy as np
import torch
from transformers import pipeline

if TYPE_CHECKING:
    from scope.core.pipelines.interface import Pipeline
else:
    Pipeline = object  # type: ignore[misc,assignment]

from scope_depth.calibration import (
    CalibrationDisplay,
    CalibrationStorage,
    CheckerboardSequence,
    GrayCodeEncoder,
    ProCamCalibrator,
    ProCamCalibration,
    StructuredLightSequence,
)
from scope_depth.pipelines.schema import DepthWebcamConfig


class Requirements:
    """Pipeline resource requirements."""

    def __init__(self, input_size: int = 1) -> None:
        """Initialize requirements."""
        self.input_size = input_size


class DepthWebcamPipeline:
    """Real-time depth estimation with ProCam calibration.

    This pipeline performs:
    1. Camera intrinsics calibration using checkerboard
    2. Structured light (Gray code) for dense correspondences
    3. Stereo calibration for camera-projector geometry
    4. Real-time depth warping to projector perspective

    Attributes:
        config: The pipeline configuration.
        device: Torch device for inference.
        depth_pipeline: HuggingFace transformers pipeline.
        procam_calibrator: ProCamCalibrator instance.
        calibration_data: Loaded ProCamCalibration or None.
        display: CalibrationDisplay for fullscreen patterns.
    """

    def __init__(self) -> None:
        """Initialize the pipeline."""
        self.config = DepthWebcamConfig()
        self.device: torch.device | None = None
        self._depth_pipeline: Any = None

        # Calibration components
        self._projector_size: tuple[int, int] = (1920, 1080)
        self.procam_calibrator: ProCamCalibrator | None = None
        self.calibration_storage = CalibrationStorage()
        self.calibration_data: ProCamCalibration | None = None
        self.display: CalibrationDisplay | None = None

        # Calibration state
        self._calibration_step: str = "none"  # none, camera, structured_light
        self._checkerboard_sequence: CheckerboardSequence | None = None
        self._structured_light_sequence: StructuredLightSequence | None = None
        self._temp_camera_images: List[np.ndarray] = []

    @classmethod
    def get_config_class(cls) -> type[DepthWebcamConfig]:
        """Return the configuration class."""
        return DepthWebcamConfig

    def _parse_resolution(self, resolution_str: str) -> tuple[int, int]:
        """Parse resolution string to (width, height)."""
        try:
            parts = resolution_str.lower().split("x")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass
        return (1920, 1080)

    def _load_calibration(self) -> None:
        """Load existing calibration if available."""
        self.calibration_data = self.calibration_storage.load_calibration()

        if self.calibration_data and self.calibration_data.is_valid:
            self._projector_size = self.calibration_data.projector.image_size
            self._update_status(
                f"ProCam Calibrated - Camera: {self.calibration_data.camera.reprojection_error:.3f}px, "
                f"Stereo: {self.calibration_data.stereo_error:.3f}px"
            )
            self.config.calibration_step = "complete"
        else:
            self.calibration_data = None
            self._update_status(
                "Not calibrated - Step 1: Calibrate Camera, Step 2: Structured Light"
            )
            self.config.calibration_step = "none"

    def _update_status(self, status: str) -> None:
        """Update calibration status in config."""
        self.config.calibration_status = status

    def _start_camera_calibration(self) -> None:
        """Start Step 1: Camera intrinsics calibration."""
        self._calibration_step = "camera"
        self._temp_camera_images = []

        # Initialize calibrator
        self._projector_size = self._parse_resolution(self.config.projector_resolution)
        self.procam_calibrator = ProCamCalibrator(projector_size=self._projector_size)

        # Initialize display (shows instructions)
        self.display = CalibrationDisplay()

        try:
            self._projector_size = self.display.start_fullscreen(
                monitor_index=self.config.projector_monitor,
                on_key=lambda k: self._on_camera_key(k),
            )

            # Update calibrator with actual size
            self.procam_calibrator = ProCamCalibrator(projector_size=self._projector_size)

            # Start checkerboard sequence
            self._checkerboard_sequence = CheckerboardSequence(
                calibrator=self.procam_calibrator,
                display=self.display,
                num_positions=self.config.num_checkerboard_images,
            )

            self._checkerboard_sequence.start(
                on_capture=self._on_checkerboard_captured,
                on_complete=self._on_camera_calibration_complete,
            )

            self._update_status(
                f"Step 1/2: Camera Calibration - Show checkerboard pattern {self.config.num_checkerboard_images}x"
            )

        except Exception as e:
            self._update_status(f"Camera calibration failed: {e}")
            self._cleanup_calibration()

    def _on_checkerboard_captured(self, current: int, total: int) -> None:
        """Handle checkerboard capture."""
        self._update_status(f"Step 1/2: Captured {current}/{total} checkerboard images")

    def _on_camera_calibration_complete(self, images: List[np.ndarray]) -> None:
        """Complete camera calibration."""
        self._temp_camera_images = images

        try:
            # Calibrate camera intrinsics
            camera_intrinsics = self.procam_calibrator.calibrate_camera(
                images,
                image_size=(images[0].shape[1], images[0].shape[0]),
            )

            self._update_status(
                f"Step 1 Complete! Camera error: {camera_intrinsics.reprojection_error:.3f}px\n"
                f"Proceed to Step 2: Structured Light"
            )
            self.config.calibration_step = "camera"

            # Show completion briefly
            import time
            time.sleep(2)

        except Exception as e:
            self._update_status(f"Camera calibration failed: {e}")

        self._cleanup_display()
        self._calibration_step = "none"

    def _start_structured_light(self) -> None:
        """Start Step 2: Structured light calibration."""
        if self.procam_calibrator is None or self.procam_calibrator.camera_intrinsics is None:
            self._update_status("Error: Complete Step 1 (Camera Calibration) first")
            return

        self._calibration_step = "structured_light"

        # Initialize display
        self.display = CalibrationDisplay()

        try:
            self.display.start_fullscreen(
                monitor_index=self.config.projector_monitor,
                on_key=lambda k: self._on_sl_key(k),
            )

            # Create Gray code encoder
            gray_encoder = GrayCodeEncoder(
                self._projector_size[0],
                self._projector_size[1],
            )

            # Start structured light sequence
            self._structured_light_sequence = StructuredLightSequence(
                gray_encoder=gray_encoder,
                display=self.display,
            )

            self._structured_light_sequence.start(
                on_pattern_change=self._on_pattern_projected,
                on_complete=self._on_structured_light_complete,
            )

            self._update_status(
                f"Step 2/2: Structured Light - Projecting {len(gray_encoder.generate_patterns())} patterns"
            )

        except Exception as e:
            self._update_status(f"Structured light failed: {e}")
            self._cleanup_calibration()

    def _on_pattern_projected(self, current: int, total: int) -> None:
        """Handle pattern projection."""
        self._update_status(f"Step 2/2: Projecting pattern {current}/{total}")

    def _on_structured_light_complete(self, captured_images: List[np.ndarray]) -> None:
        """Complete structured light calibration."""
        try:
            # Add correspondences
            self.procam_calibrator.add_structured_light_correspondences(captured_images)

            # Perform stereo calibration
            procam = self.procam_calibrator.calibrate_stereo()

            # Save calibration
            self.calibration_storage.save_calibration(procam)
            self.calibration_data = procam

            self._update_status(
                f"Calibration Complete!\n"
                f"Camera error: {procam.camera.reprojection_error:.3f}px\n"
                f"Stereo error: {procam.stereo_error:.3f}px"
            )
            self.config.calibration_step = "complete"

            # Export depth map
            self.calibration_storage.export_depth_map(self._projector_size)

            import time
            time.sleep(3)

        except Exception as e:
            self._update_status(f"Stereo calibration failed: {e}")

        self._cleanup_calibration()

    def _on_camera_key(self, key: str) -> None:
        """Handle key during camera calibration."""
        if key == "s" and self._checkerboard_sequence:
            self._checkerboard_sequence.skip()
        elif key == "q":
            if self._checkerboard_sequence:
                self._checkerboard_sequence.cancel()
            self._cleanup_calibration()

    def _on_sl_key(self, key: str) -> None:
        """Handle key during structured light."""
        if key == "q":
            if self._structured_light_sequence:
                self._structured_light_sequence.cancel()
            self._cleanup_calibration()

    def _cleanup_display(self) -> None:
        """Clean up display only."""
        if self.display:
            self.display.stop()
            self.display = None

    def _cleanup_calibration(self) -> None:
        """Clean up all calibration resources."""
        self._cleanup_display()
        self._checkerboard_sequence = None
        self._structured_light_sequence = None
        self._calibration_step = "none"

    def prepare(
        self,
        config: DepthWebcamConfig,
        device: torch.device | None = None,
    ) -> Requirements:
        """Prepare the pipeline."""
        self.config = config

        # Initialize device
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        device_idx = 0 if str(self.device) != "cpu" else -1

        # Initialize depth pipeline
        self._depth_pipeline = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device_idx,
        )

        # Load existing calibration
        self._load_calibration()

        # Check calibration triggers
        if config.trigger_camera_calibration:
            config.trigger_camera_calibration = False
            self._start_camera_calibration()

        if config.trigger_structured_light:
            config.trigger_structured_light = False
            self._start_structured_light()

        if config.clear_calibration:
            self.calibration_storage.delete_calibration()
            self.calibration_data = None
            config.clear_calibration = False
            self._update_status("Calibration cleared")

        return Requirements(input_size=1)

    def __call__(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Process video frame and return depth map."""
        if self._depth_pipeline is None:
            raise RuntimeError("Pipeline not prepared")

        # Handle calibration steps
        if self._calibration_step == "camera" and self._checkerboard_sequence:
            return self._process_camera_calibration_frame(video)

        if self._calibration_step == "structured_light" and self._structured_light_sequence:
            return self._process_structured_light_frame(video)

        # Normal depth processing
        return self._process_depth(video)

    def _process_camera_calibration_frame(
        self,
        video: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process frame during camera calibration."""
        # Convert to numpy
        frame = video[0].cpu().numpy().astype(np.uint8)

        # Update display
        if self.display:
            self.display.update()

        # Try to capture checkerboard
        success = self._checkerboard_sequence.attempt_capture(frame)

        # Return camera feed (normalized)
        return {"video": video / 255.0}

    def _process_structured_light_frame(
        self,
        video: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process frame during structured light."""
        frame = video[0].cpu().numpy().astype(np.uint8)

        # Update display
        if self.display:
            self.display.update()

        # Capture frame for current pattern
        self._structured_light_sequence.capture_frame(frame)

        return {"video": video / 255.0}

    def _process_depth(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        """Process depth estimation and warp to projector perspective."""
        frame = video[0]

        # Convert to PIL
        from PIL import Image

        frame_np = frame.cpu().numpy().astype("uint8")
        pil_image = Image.fromarray(frame_np)

        # Run depth estimation
        result = self._depth_pipeline(pil_image)
        depth_image = result["depth"]

        # Convert to tensor
        depth_array = torch.from_numpy(np.array(depth_image, dtype=np.float32))

        # Normalize
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        if depth_max > depth_min:
            depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = torch.zeros_like(depth_array)

        # Apply depth scale
        depth_normalized = torch.clamp(
            depth_normalized * self.config.depth_scale, 0.0, 1.0
        )

        # Convert to 3-channel
        depth_rgb = depth_normalized.unsqueeze(-1).repeat(1, 1, 3)

        # Apply ProCam calibration if available
        if (
            self.config.use_calibration
            and self.calibration_data is not None
            and self.calibration_data.is_valid
        ):
            depth_rgb = self._warp_to_projector(depth_rgb, frame_np)

        # Add batch dimension
        output = depth_rgb.unsqueeze(0)

        return {"video": output}

    def _warp_to_projector(
        self,
        depth_rgb: torch.Tensor,
        camera_frame: np.ndarray,
    ) -> torch.Tensor:
        """Warp depth from camera to projector perspective.

        Uses the full ProCam calibration with:
        1. Camera undistortion
        2. Stereo rectification
        3. Reprojection to projector
        """
        # Convert tensor to numpy
        depth_np = depth_rgb.cpu().numpy()

        # Step 1: Undistort camera image using calibrated intrinsics
        undistorted = self.calibration_data.undistort_camera_image(
            (depth_np * 255).astype(np.uint8)
        )

        # Step 2: Warp to projector perspective
        warped = self.calibration_data.warp_camera_to_projector(
            undistorted,
            output_size=self.calibration_data.projector.image_size
            if self.config.output_to_projector_resolution
            else None,
        )

        # Convert back to tensor
        warped_tensor = torch.from_numpy(warped.astype(np.float32) / 255.0)

        # Ensure same device and dtype
        return warped_tensor.to(depth_rgb.device, dtype=depth_rgb.dtype)


# Register as Pipeline for type checking
if not TYPE_CHECKING:
    DepthWebcamPipeline.__bases__ = (Pipeline,)
