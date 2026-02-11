"""Configuration schema for the Depth Webcam Pipeline with ProCam calibration."""

from pydantic import BaseModel, Field


class ModeDefaults(BaseModel):
    """Default mode configuration for a pipeline mode."""

    default: bool = True


class DepthWebcamConfig(BaseModel):
    """Configuration schema for the Depth Webcam Pipeline.

    This schema defines the configuration parameters exposed in the Scope UI,
    including pipeline identification, depth adjustment parameters, and
    ProCam calibration settings for projector-camera alignment.
    """

    pipeline_id: str = "projection-map-anything"
    pipeline_name: str = "ProjectionMapAnything VJ.Tools"
    supports_prompts: bool = False
    modes: dict[str, ModeDefaults] = {"video": ModeDefaults(default=True)}

    # Depth estimation parameters
    depth_scale: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Scale factor to adjust depth map intensity",
    )

    # Projector Calibration Parameters
    projector_monitor: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Monitor index for projector output (0=primary, 1=secondary)",
    )

    projector_resolution: str = Field(
        default="1920x1080",
        description="Projector resolution as WIDTHxHEIGHT (e.g., 1920x1080)",
    )

    # Calibration workflow control
    calibration_step: str = Field(
        default="none",
        description="Current calibration step: none, camera, structured_light, complete",
    )

    trigger_camera_calibration: bool = Field(
        default=False,
        description="Step 1: Calibrate camera intrinsics using checkerboard",
    )

    trigger_structured_light: bool = Field(
        default=False,
        description="Step 2: Project Gray codes to find camera-projector correspondences",
    )

    clear_calibration: bool = Field(
        default=False,
        description="Clear all calibration data and start over",
    )

    calibration_status: str = Field(
        default="Not calibrated - Complete both calibration steps",
        description="Current calibration status (read-only)",
    )

    # Advanced options
    num_checkerboard_images: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Number of checkerboard images for camera calibration",
    )

    use_calibration: bool = Field(
        default=True,
        description="Apply projector calibration warp to depth output",
    )

    # Output settings
    output_to_projector_resolution: bool = Field(
        default=True,
        description="Output depth at projector resolution (vs camera resolution)",
    )
