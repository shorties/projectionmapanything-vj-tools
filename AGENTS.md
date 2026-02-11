# Scope Depth Plugin - Agent Reference

## Project Overview

This is a **Daydream Scope plugin** that performs real-time monocular depth estimation using the Depth Anything V2 model. It captures frames from a standard UVC-compatible webcam, runs depth prediction on each frame, and outputs a video stream of depth maps.

### Key Features

- **Webcam Input**: Supports any UVC-compatible webcam as the video source
- **Depth Estimation**: Uses Depth Anything V2 Small model (Apache-2.0 licensed) for per-frame depth prediction
- **Real-Time Pipeline**: Low-latency processing integrated with Scope's WebRTC streaming
- **Calibration UI**: Depth Scale slider for adjusting depth map intensity (range 0.1-5.0, default 1.0)
- **Compatibility**: Works in local Scope installations and on cloud GPU instances (e.g., RunPod)

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12+ |
| Framework | Daydream Scope Plugin System (Pluggy-based hooks) |
| ML Framework | PyTorch + Hugging Face Transformers |
| Model | Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) |
| Build System | Hatchling |
| Package Manager | uv (used by Scope) |

## Project Structure

```
scope-depth/
├── pyproject.toml              # Package metadata, entry point, dependencies
├── README.md                   # Human-readable documentation
├── clarification-harvest/      # Development skill for requirement gathering
│   ├── SKILL.md
│   └── references/
│       ├── database-questions.md
│       ├── deployment-questions.md
│       └── web-api-questions.md
├── clarification-harvest.skill # Skill file reference
└── src/scope_depth/            # Main package
    ├── __init__.py             # Package version (0.1.0)
    ├── plugin.py               # Pluggy hook implementation (register_pipelines)
    └── pipelines/
        ├── __init__.py         # Exports DepthWebcamConfig, DepthWebcamPipeline
        ├── schema.py           # DepthWebcamConfig (Pydantic model)
        └── pipeline.py         # DepthWebcamPipeline (Depth Anything V2 inference)
```

### Key Source Files

| File | Purpose |
|------|---------|
| `plugin.py` | Implements `register_pipelines(register)` hook to register the pipeline class with Scope |
| `pipelines/schema.py` | Configuration schema with UI parameters (depth_scale slider) |
| `pipelines/pipeline.py` | Main pipeline logic (DepthWebcamPipeline) - handles model loading, frame processing, depth estimation |

## Architecture Details

### Plugin Registration

Scope plugins are discovered via Python entry points in `pyproject.toml`:

```toml
[project.entry-points."scope"]
scope_depth = "scope_depth.plugin"
```

The plugin implements the `register_pipelines(register)` hook to register the pipeline class.

### Pipeline Class Contract

The `DepthWebcamPipeline` class must:

1. Inherit from `scope.core.pipelines.interface.Pipeline`
2. Implement `get_config_class()` returning the config schema (`DepthWebcamConfig`)
3. Implement `prepare(config, device)` returning `Requirements(input_size=1)`
4. Implement `__call__(video)` to process frames

### Configuration Schema

Key configuration fields in `DepthWebcamConfig`:

```python
pipeline_id = "depth-webcam"
pipeline_name = "Depth (Webcam)"
supports_prompts = False
modes = {"video": ModeDefaults(default=True)}

# UI Parameter
depth_scale: float = Field(
    default=1.0, ge=0.1, le=5.0,
    description="Scale factor to adjust depth map intensity"
)
```

### Frame Processing Flow

1. **Input**: Video frame as torch tensor (shape `(1, H, W, C)`, values 0-255)
2. **Conversion**: Convert to PIL Image for the model
3. **Inference**: Run Depth Anything V2 model via HuggingFace pipeline
4. **Post-processing**:
   - Convert depth output to tensor
   - Min-max normalize to [0,1] range
   - Apply depth_scale calibration
   - Convert to 3-channel RGB
5. **Output**: Return `{"video": depth_tensor}` (shape `(1, H, W, 3)`, values [0,1])

## Build and Development

### Installation Commands

```bash
# Development installation (editable mode)
uv run daydream-scope install -e /path/to/scope-depth

# Production installation from Git URL
uv run daydream-scope install https://github.com/YourUser/scope-depth.git
```

### Dependencies

Core dependencies (from `pyproject.toml`):

```toml
[project.dependencies]
transformers = ">=4.33.0"
```

Development dependencies:
- pytest>=7.0.0
- ruff>=0.1.0
- mypy>=1.0.0

Scope provides: PyTorch, FastAPI, Pydantic, and other core packages.

### Build System

Uses Hatchling as the build backend:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Code Style Guidelines

Configured via `pyproject.toml`:

### Ruff (Linting & Formatting)
- Line length: 100 characters
- Target Python version: 3.12
- Enabled rules: E, F, I, W, N, D, UP
- Ignored rules: D100, D104, D107 (missing docstrings)

### MyPy (Type Checking)
- Python version: 3.12
- Strict mode: `warn_return_any`, `warn_unused_configs`, `disallow_untyped_defs`

### Coding Conventions

- Follow PEP 8 conventions
- Use type hints (Python 3.12+ features supported)
- Import from `scope.core.*` for Scope integration
- Use `TYPE_CHECKING` for circular import prevention

Example pattern from codebase:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scope.core.pipelines.interface import Pipeline
else:
    Pipeline = object  # type: ignore[misc,assignment]
```

## Testing Instructions

### Local Testing

1. Launch Scope with GPU access:
   ```bash
   uv run daydream-scope
   ```

2. Install the plugin (development or production mode)

3. Select "Depth (Webcam)" pipeline from the dropdown

4. Connect webcam feed in the Input settings

5. Click Play ▶️ to start streaming

6. Adjust "Depth Scale" slider (0.1-5.0) in Settings panel for calibration

### Validation Criteria

- Verify relative depth ordering (closer objects should differ in intensity from farther ones)
- Check for dropped frames or high latency
- Target: Several FPS on mid-range GPU, ~30 FPS on high-end GPUs

### Cloud Deployment (RunPod)

1. Use Docker image: `daydreamlive/scope:main`
2. Enable SSH/HTTP tunnel for UI access
3. Install plugin via CLI or UI
4. Note: Cloud instances don't have webcams - test with video files or IP cameras

## Device Handling

```python
# Initialize device from Scope
self.device = device if device is not None else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Pass to HF pipeline
device=0 if str(self.device) != "cpu" else -1
```

## Security Considerations

- No API keys or credentials required for basic functionality
- Model weights are downloaded from HuggingFace on first run
- Plugin operates entirely locally (no external API calls during inference)
- Cloud deployment requires self-hosted Scope (official Scope Cloud Inference beta does NOT support custom plugins)

## License Notes

- **Plugin Code**: MIT License
- **Depth Anything V2 Small**: Apache-2.0 (commercial use allowed)
- **Base/Large variants**: Some are CC BY-NC (non-commercial) - avoid using these

## Future Enhancement Areas

Based on the specification document (`AutoProjectionMap.md`), potential future improvements include:

- Temporal smoothing for frame consistency
- Metric depth variant support (`Depth-Anything-V2-Metric-Outdoor-Small-hf`)
- Colormap visualization option (instead of grayscale)
- Integration with Scope's VACE system as a preprocessor
- Performance optimization using AutoModel API directly to avoid PIL round-trip

## References

- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- HuggingFace Model: `depth-anything/Depth-Anything-V2-Small-hf`
- Detailed Specification: `AutoProjectionMap.md`
- Clarification Harvest Skill: `clarification-harvest/SKILL.md`
