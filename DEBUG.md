# Debugging Plugin Not Showing in Scope

## How It Should Work

Your plugin registers ONE pipeline that appears in the dropdown:
- **Pipeline Name**: "ProjectionMapAnything VJ.Tools" 
- **Pipeline ID**: "projection-map-anything"

## Common Reasons It's Not Showing

### 1. Plugin Crashing During Load

Scope silently skips plugins that fail to import. Check for errors:

```bash
# Test if the plugin imports without errors
uv run python -c "from scope_depth.plugin import register_pipelines; print('OK: plugin imports')"

# Test the full pipeline
uv run python -c "
from scope_depth.pipelines import DepthWebcamPipeline
p = DepthWebcamPipeline()
config = p.get_config_class()()
print(f'Pipeline: {config.pipeline_name}')
print(f'ID: {config.pipeline_id}')
"
```

### 2. Missing Dependencies

The container might be missing OpenCV or other deps:

```bash
# Check if all deps are installed
uv run python -c "import cv2; import transformers; import PIL; print('All deps OK')"

# If missing, install them
uv pip install opencv-python transformers pillow
```

### 3. Entry Point Not Registered

Check if Scope can see the entry point:

```bash
# Should show: ['scope_depth']
uv run python -c "
from importlib.metadata import entry_points
eps = entry_points(group='scope')
print([e.name for e in eps])
"
```

### 4. Install in Editable Mode (Development)

For local development, install in editable mode so changes are immediate:

```bash
cd /root/projectionmapanything-vj-tools
uv pip install -e .

# Or for Scope's environment specifically:
uv run pip install -e /root/projectionmapanything-vj-tools
```

## Quick Fix to Try

```bash
# 1. Clear any broken state
rm -rf /root/.daydream-scope/plugins/

# 2. Install dependencies manually first
uv pip install opencv-python transformers pillow

# 3. Clone fresh
cd /root
git clone https://github.com/shorties/projectionmapanything-vj-tools.git

# 4. Install in editable mode
uv pip install -e /root/projectionmapanything-vj-tools

# 5. Check if entry point exists
uv run python -c "from importlib.metadata import entry_points; eps = entry_points(group='scope'); print('Entry points:', [e.name for e in eps])"

# 6. Restart Scope completely
# (Stop and restart the daydream-scope process)
```

## Check Scope Logs

When you start Scope, watch for error messages:

```bash
uv run daydream-scope 2>&1 | grep -i "error\|scope_depth\|projection"
```

## Does the Pipeline Need a Model?

Your pipeline DOES include the model - it downloads `depth-anything/Depth-Anything-V2-Small-hf` from HuggingFace on first run. That's handled by the `transformers` library in `pipeline.py`.

The model is NOT bundled with the plugin - it's downloaded automatically (~100MB).

## Expected Behavior

After successful installation:
1. Start Scope
2. Click pipeline dropdown
3. See "ProjectionMapAnything VJ.Tools" in the list
4. Select it
5. Settings panel shows calibration buttons

If you don't see it in step 3, the plugin failed to load during startup.
