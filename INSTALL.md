# Installation Guide for ProjectionMapAnything VJ.Tools

## Prerequisites

- Python 3.12+
- Daydream Scope installed
- uv package manager (comes with Scope)

## Step-by-Step Installation

### 1. Navigate to Project Directory

```bash
cd D:\Github\AutoProjectionMap
```

### 2. Install in Editable Mode

This registers the entry point so Scope can discover the plugin:

```bash
# Install the package in editable mode
uv pip install -e .

# Or if Scope has its own environment:
uv run pip install -e .
```

### 3. Verify Installation

Check that the entry point is registered:

```bash
uv run python -c "from importlib.metadata import entry_points; eps = entry_points(); scope_eps = eps.get('scope', []); print([e.name for e in scope_eps])"
```

You should see: `['scope_depth']`

### 4. Restart Daydream Scope

Close Scope completely and reopen it. The pipeline should now appear.

### 5. Check Scope Logs (if not appearing)

If the pipeline still doesn't appear, check Scope's logs for import errors:

```bash
# Look for Scope's log output in the terminal where you launched it
uv run daydream-scope
```

Watch for any error messages about `scope_depth` during startup.

## Debugging

### Check if plugin loads manually

```python
# Run this in the Scope environment
uv run python -c "
from scope_depth.plugin import register_pipelines
from scope_depth.pipelines import DepthWebcamPipeline

print('Plugin loads successfully')
print('Pipeline name:', DepthWebcamPipeline.get_config_class()().pipeline_name)
"
```

### Common Issues

1. **"No module named 'scope_depth'"**
   - The package isn't installed. Run `uv pip install -e .`

2. **"No module named 'cv2'" or "transformers"**
   - Dependencies missing. Run `uv pip install opencv-python transformers pillow`

3. **Pipeline doesn't appear in dropdown**
   - Entry point not registered. Verify with: `uv run pip show projectionmapanything-vj-tools`
   - Try reinstalling: `uv run pip uninstall projectionmapanything-vj-tools && uv pip install -e .`

4. **ImportError during Scope startup**
   - Check that all dependencies are installed in Scope's environment
   - Look at the console output when starting Scope

## Uninstall

```bash
uv pip uninstall projectionmapanything-vj-tools
```

Or remove the entry point manually by editing Scope's plugin configuration.
