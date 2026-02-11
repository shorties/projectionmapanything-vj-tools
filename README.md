# ProjectionMapAnything VJ.Tools

A Daydream Scope plugin for real-time monocular depth estimation using Depth Anything V2, with full ProCam calibration for projection mapping VJ applications.

## Features

- **Webcam Input**: Supports any UVC-compatible webcam
- **Depth Estimation**: Uses Depth Anything V2 Small model (Apache-2.0)
- **ProCam Calibration**: Full camera and projector intrinsic calibration with stereo extrinsics
- **Lens Distortion Correction**: Proper undistortion using calibrated camera matrix
- **Structured Light**: Gray code patterns for dense camera-projector correspondences
- **Real-Time Warping**: Maps camera depth to projector perspective on GPU
- **ControlNet Export**: Export calibration data for ControlNet integration

## Installation

### Requirements

- Python 3.12+
- Daydream Scope
- Webcam
- Projector connected to PC
- Checkerboard pattern (printed or displayed on tablet/phone)

### Dependencies

```bash
pip install transformers>=4.33.0 opencv-python>=4.8.0 pillow>=10.0.0
```

### Install Plugin

```bash
# From GitHub (tarball - recommended)
uv run daydream-scope install "https://github.com/shorties/projectionmapanything-vj-tools/archive/refs/heads/main.tar.gz"

# Development mode (local)
uv run daydream-scope install -e /path/to/projectionmapanything-vj-tools
```

## Usage

### Basic Setup

1. Launch Daydream Scope: `uv run daydream-scope`
2. Select "ProjectionMapAnything VJ.Tools" pipeline
3. Connect webcam in Input settings

### ProCam Calibration

The calibration has **two steps** that must be completed in order:

#### Step 1: Camera Intrinsics Calibration

Calibrate your camera's lens distortion and intrinsic parameters.

1. Print or display a checkerboard pattern (6×9 inner corners recommended)
2. In plugin settings, click **"Step 1: Calibrate Camera"**
3. A fullscreen window appears on the projector display
4. **Show the checkerboard to the camera** from various angles
5. The system automatically captures when the pattern is detected
6. Capture **15 different positions** (configurable)
7. Step 1 completes automatically

**Tips:**
- Move the checkerboard to different positions and angles
- Ensure good lighting on the checkerboard
- Fill the camera frame with the pattern
- Include some close-up and some far-away positions

#### Step 2: Structured Light Calibration

Find dense correspondences between camera and projector using Gray codes.

1. Ensure the projector is displaying a visible area
2. In plugin settings, click **"Step 2: Structured Light"**
3. The projector displays a sequence of binary patterns (Gray codes)
4. **Keep the camera still** during this process
5. Wait for all patterns to project and capture (~30 seconds)
6. Calibration completes automatically

**Tips:**
- Ensure projector is in focus and bright enough
- Minimize ambient light
- Don't move camera or projector during capture
- The projected patterns should fill the camera's view

### Using the Calibrated Pipeline

1. After both steps complete, the status shows:  
   `ProCam Calibrated - Camera: 0.123px, Stereo: 0.456px`
2. Click **Play ▶️** to start streaming
3. Depth output is automatically warped to projector perspective
4. Adjust "Depth Scale" to tune depth intensity

### Configuration

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Depth Scale | 0.1 - 5.0 | 1.0 | Depth intensity multiplier |
| Projector Monitor | 0 - 10 | 0 | Display index for projector |
| Projector Resolution | "WxH" | "1920x1080" | Projector native resolution |
| Num Checkerboard Images | 5 - 50 | 15 | Camera calibration captures |
| Use Calibration | bool | true | Enable projector warping |
| Output to Projector Resolution | bool | true | Match output to projector |

### Calibration Files

Calibration is saved to `procam_calibration.json`:

```json
{
  "camera_K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "camera_dist_coeffs": [k1, k2, p1, p2, k3],
  "projector_K": [...],
  "R": [[...]],  // Camera to projector rotation
  "T": [...],    // Camera to projector translation
  "stereo_error": 0.456
}
```

To share calibration between machines, copy this file.

## How It Works

### ProCam Calibration Pipeline

Unlike simple homography-based approaches, this implements full **ProCam calibration** similar to RoomAlive Toolkit:

```
┌─────────────────┐
│  Camera Images  │  Checkerboard captures
│  (Step 1)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  cv2.calibrate  │  Camera intrinsics (K, distortion)
│  Camera         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gray Code      │  Structured light patterns
│  Patterns       │
│  (Step 2)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Decode         │  Dense correspondences
│  Correspondences│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  cv2.stereo     │  Camera & projector intrinsics
│  Calibrate      │  + R, T (relative pose)
└─────────────────┘
```

### Depth Warping

For each frame, the pipeline:

1. **Estimate Depth**: Depth Anything V2 → raw depth map
2. **Undistort**: Remove camera lens distortion using calibrated K/dist_coeffs
3. **Rectify**: Apply stereo rectification using R, T
4. **Reproject**: Map to projector image plane using projector K
5. **Output**: Depth in projector perspective

This accounts for:
- Camera lens distortion (radial, tangential)
- Projector lens distortion
- Perspective differences between camera and projector
- Non-planar alignment

## Architecture

```
scope_depth/
├── __init__.py
├── plugin.py
├── pipelines/
│   ├── __init__.py
│   ├── schema.py           # UI configuration
│   └── pipeline.py         # Main pipeline logic
└── calibration/
    ├── __init__.py
    ├── calibrator.py       # ProCamCalibrator, GrayCodeEncoder
    ├── display.py          # Fullscreen pattern display
    └── storage.py          # JSON persistence
```

### Key Classes

- **ProCamCalibrator**: Main calibration orchestrator
- **GrayCodeEncoder**: Generates structured light patterns
- **ProCamCalibration**: Stores full camera-projector geometry
- **CalibrationDisplay**: Tkinter fullscreen display

## Troubleshooting

### "Camera calibration failed"
- Ensure checkerboard is clearly visible
- Need at least 3 successful captures
- Check pattern is 6×9 inner corners
- Improve lighting on checkerboard

### "Too few correspondences" (Structured Light)
- Increase projector brightness
- Reduce ambient light
- Move camera closer to projection
- Check projector focus

### Depth misalignment after calibration
- Verify calibration errors are < 1.0 px
- Redo calibration with better captures
- Ensure camera/projector didn't move between steps
- Check projector resolution setting matches actual

### Low FPS
- ProCam warping adds ~5-10ms per frame
- Disable "Output to Projector Resolution" for faster processing
- Reduce depth model size if needed

## Future Enhancements

- [ ] Real-time depth camera integration (Kinect, RealSense)
- [ ] Multi-projector support
- [ ] NDI output for network projectors
- [ ] Bundle adjustment refinement
- [ ] Automatic recalibration detection
- [ ] Calibration quality visualization

## References

- **RoomAlive Toolkit**: https://github.com/Microsoft/RoomAliveToolkit
- **Depth Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2
- **OpenCV Calibration**: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- **Gray Codes**: https://en.wikipedia.org/wiki/Gray_code

## License

- Plugin: MIT License
- Depth Anything V2: Apache-2.0
