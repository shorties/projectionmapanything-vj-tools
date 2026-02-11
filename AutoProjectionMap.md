VJ.Tools AutoProjectionMap Daydream Scope Plugin Specification


Overview



We aim to develop a Daydream Scope plugin that uses any standard webcam feed to perform real-time monocular depth estimation using the Depth Anything V2 model. This plugin will capture frames from a connected webcam, run a depth prediction model on each frame, and output a video stream of depth maps (visualizing closer vs. farther objects). Daydream Scope’s plugin system will handle the UI and streaming, while our plugin provides the custom pipeline logic. The Depth Anything V2 model is open-source and Apache-2.0 licensed, ensuring we can integrate it freely.



Key Features and Goals:



Webcam Input: Support any UVC-compatible webcam as the video source. Daydream Scope allows video pipelines to use camera feeds as input, so the plugin will process live frames from the user’s webcam.



Depth Estimation Model: Utilize Depth Anything V2 (preferably the Small variant for efficiency) for per-frame depth prediction. This model produces a dense depth map from a single image and is significantly faster and more detailed than previous approaches.



Real-Time Pipeline: Integrate with Scope’s real-time streaming – the plugin will output a synchronized depth video in low latency. Scope’s architecture streams AI-generated video via WebRTC for interactive experiences, and our pipeline will fit into that framework.



UI Controls (Calibration): Provide UI settings for simple calibration or tuning. For example, a Depth Scale slider can allow adjusting the depth map intensity to aid calibration (e.g. aligning relative depth to approximate real distances). If more precise metric depth is needed in the future, we can accommodate it by using fine-tuned metric models or user-provided reference distances. (Depth Anything V2 can be fine-tuned on metric data like NYUv2/KITTI for absolute depth, but by default it predicts relative depth.)



Compatibility: The plugin should work in local Scope installations and on cloud GPU instances (e.g. RunPod). Note that Daydream’s official Remote Inference beta does not support plugins, so running this plugin on the cloud will require a self-hosted Scope server (for example, on RunPod via a tunnel). We will ensure instructions for deploying in such environments are provided.



Finally, note that Scope already includes a built-in depth preprocessor called “video-depth-anything” for V2V control workflows. Our plugin will create a standalone pipeline that directly outputs the depth video (useful for visualization or further processing), potentially with added calibration features beyond the built-in preprocessor.



Daydream Scope Plugin Architecture



Scope plugins are Python packages that register one or more pipelines with the Scope application. A pipeline is a class defining a generative or processing workflow for video frames and/or prompts. The Scope host handles plugin discovery, model loading, UI controls, and streaming, while the plugin provides the custom processing logic. Key points about the plugin system:



Entry Point Registration: The plugin is discovered via a Python entry point in its pyproject.toml. We will specify an entry under the "scope" group pointing to our plugin module (e.g. scope\_depth = "scope\_depth.plugin"), so Scope can find our code.



Hook Implementation: Scope uses the Pluggy hook system. Our plugin must implement the register\_pipelines(register) hook to inform Scope about our pipeline class. Scope calls this at startup to add our pipeline to its registry.



Pipeline Class Contract: Each pipeline class must declare a configuration schema (for UI controls), the input/output modalities it supports, and implement a \_\_call\_\_ method to process frames (and optionally a prepare method). Scope will handle reading from the webcam and passing frames to \_\_call\_\_, then streaming out the returned results. The pipeline doesn’t manage any GUI or device I/O directly – it just processes data frames.



Video Mode Pipeline: Our pipeline will operate in video-only mode since it requires a continuous video input (webcam) and no text prompt. We will indicate this by setting the pipeline mode to "video" only. In practice, this means Scope will supply video frames (from either a camera feed or video file) to our pipeline.



Scope UI Integration: Scope automatically generates UI panels for pipeline parameters and allows switching pipelines. Our plugin will appear in the pipeline selection dropdown once installed. Any parameters defined in the pipeline’s config schema (with ui\_field\_config) will be rendered as sliders, checkboxes, etc., in the Settings or Input panel. For example, we can expose a “Depth Scale” slider for calibration – Scope will show it in the UI and pass its value into our pipeline on each frame.



Device and Performance: The Scope environment provides PyTorch and runs pipelines on the designated device (GPU if available). We will write our pipeline to move data to the correct device and leverage the GPU for the depth model. Scope passes a device argument when instantiating the pipeline class, and we should use it. The Depth Anything V2 Small model (~25M params) is lightweight (around 1 GB VRAM usage for real-time depth) and can achieve interactive frame rates on modern GPUs. We’ll ensure only the necessary frames are processed (likely one frame at a time) to minimize latency.



Project Setup and Installation



We will structure the plugin as a standalone Python package. The following layout is recommended:



scope-depth/

├── pyproject.toml

└── src/

&nbsp;   └── scope\_depth/

&nbsp;       ├── \_\_init\_\_.py

&nbsp;       ├── plugin.py

&nbsp;       ├── pipelines/

&nbsp;       │   ├── \_\_init\_\_.py

&nbsp;       │   ├── schema.py

&nbsp;       │   └── pipeline.py

&nbsp;       └── ... (any additional modules)





pyproject.toml – defines the package metadata, entry point, and dependencies. Key sections in this file:



Project Metadata: Name the package (e.g. "scope-depth") and set the Python requirement (Scope requires Python 3.12+).



Entry Point: Under \[project.entry-points."scope"], add an entry for our plugin. For example:



\[project]

name = "scope-depth"

version = "0.1.0"

description = "Depth estimation plugin for Daydream Scope"

requires-python = ">=3.12"



\[project.entry-points."scope"]

scope\_depth = "scope\_depth.plugin"





This tells Scope to import scope\_depth/plugin.py as our plugin entry. The key (scope\_depth) is the plugin name shown in Scope’s UI.



Dependencies: List any third-party libraries our plugin needs that are not already included with Scope. Scope’s environment provides many core packages (PyTorch, FastAPI, Pydantic, etc.), but we will need the Hugging Face Transformers library for the depth model. We add:



\[project.dependencies]

transformers = ">=4.33.0"





This ensures Scope will install Transformers (version 4.33+ which supports Depth Anything V2) when our plugin is installed. (Scope uses the uv package manager to resolve and install plugin deps automatically.)



plugin.py – defines the hook implementation. We simply import our pipeline class and register it:



\# src/scope\_depth/plugin.py

from scope.core.plugins.hookspecs import hookimpl



@hookimpl

def register\_pipelines(register):

&nbsp;   from scope\_depth.pipelines.pipeline import DepthWebcamPipeline

&nbsp;   register(DepthWebcamPipeline)





This uses Scope’s register\_pipelines hook to add our pipeline into the system. The register() function is provided by Scope – we call it for each pipeline class we want to expose. In our case, just one pipeline (DepthWebcamPipeline). After installation, Scope will call this and map a pipeline ID to our class.



With these in place, installing the plugin is straightforward. During development, one can use the CLI: uv run daydream-scope install -e /path/to/scope-depth to install in editable mode. For end users, providing a Git URL is easiest. For example, if our plugin is on GitHub, a user could enter the repo URL in Scope’s Plugins settings or run:



uv run daydream-scope install https://github.com/YourUser/scope-depth.git





Scope will fetch the package, install dependencies (like transformers), and restart the server to load the plugin. After that, the plugin’s pipeline appears in the UI pipeline selector.



Pipeline Configuration Schema (schema.py)



The pipeline’s schema class defines the ID, name, description, and any adjustable parameters for our depth pipeline. It inherits from BasePipelineConfig (which provides common fields and validation). For our DepthWebcamPipeline, the schema might look like:



\# src/scope\_depth/pipelines/schema.py

from pydantic import Field

from scope.core.pipelines.base\_schema import BasePipelineConfig, ModeDefaults, ui\_field\_config



class DepthWebcamConfig(BasePipelineConfig):

&nbsp;   """Configuration for the Depth Webcam pipeline."""

&nbsp;   pipeline\_id = "depth-webcam"  

&nbsp;   pipeline\_name = "Depth (Webcam)"

&nbsp;   pipeline\_description = "Real-time depth estimation from any webcam feed"



&nbsp;   supports\_prompts = False  # No text prompts needed

&nbsp;   modes = {"video": ModeDefaults(default=True)}  # Requires a video input (webcam)



&nbsp;   # Example UI parameter: a calibration scale slider for depth intensity

&nbsp;   depth\_scale: float = Field(

&nbsp;       default=1.0, ge=0.1, le=5.0,

&nbsp;       description="Scale factor to adjust depth map intensity (calibration)",

&nbsp;       json\_schema\_extra=ui\_field\_config(order=1, label="Depth Scale")

&nbsp;   )





Key details:



We assign a unique pipeline\_id (used internally and in CLI commands) and human-friendly name/description for the UI.



supports\_prompts = False indicates this pipeline doesn’t use text prompts at all (only visual input).



modes = {"video": ModeDefaults(default=True)} declares that this pipeline only operates in video mode. This means Scope will require a video source (webcam or video file) to run it. The ModeDefaults(default=True) just marks this as the default mode for the pipeline (since there is only one mode here).



We added a depth\_scale field as a runtime parameter to help with calibration. This will show up as a slider in the Scope UI (labeled "Depth Scale"). It’s defined with a range 0.1–5.0 and a default of 1.0 (no scaling). We use ui\_field\_config to configure its appearance – here we just set an order and label. This parameter is not marked as is\_load\_param, so it’s adjustable during streaming (Scope will send its value on each frame). The idea is that the user can brighten or dim the depth map output to roughly calibrate or enhance contrast. (For example, if the raw depth appears too faint or if the user knows the scene’s actual scale, they might increase this factor so farther distances aren’t all clipped to dark.)



We can add more parameters in the future (e.g., a toggle for using a metric-trained model vs relative model, or an option to output a colorized depth map). For now, we keep the config minimal. The fields defined here are accessible in the pipeline’s \_\_call\_\_ via kwargs at runtime.



Pipeline Implementation (pipeline.py)



Now we implement the actual pipeline logic in the DepthWebcamPipeline class. This class inherits from scope.core.pipelines.interface.Pipeline and must define how many frames it needs and how to process them. Below is a possible implementation with detailed explanations:



\# src/scope\_depth/pipelines/pipeline.py

from typing import TYPE\_CHECKING

import torch

import numpy as np

from PIL import Image

from transformers import pipeline as hf\_pipeline



from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import DepthWebcamConfig



if TYPE\_CHECKING:

&nbsp;   from scope.core.pipelines.base\_schema import BasePipelineConfig



class DepthWebcamPipeline(Pipeline):

&nbsp;   """Pipeline that computes depth maps from webcam video frames using Depth Anything V2."""

&nbsp;   

&nbsp;   @classmethod

&nbsp;   def get\_config\_class(cls) -> type\["BasePipelineConfig"]:

&nbsp;       return DepthWebcamConfig  # Link to our config schema

&nbsp;   

&nbsp;   def \_\_init\_\_(self, device: torch.device | None = None, \*\*kwargs):

&nbsp;       # Determine device (GPU or CPU) for model execution

&nbsp;       self.device = device if device is not None else torch.device("cuda" if torch.cuda.is\_available() else "cpu")

&nbsp;       # Initialize the depth estimation model/pipeline

&nbsp;       # We use Hugging Face transformers pipeline for depth estimation:contentReference\[oaicite:49]{index=49}

&nbsp;       # 'depth-anything/Depth-Anything-V2-Small-hf' is the small DA V2 model (Apache-2.0 licensed):contentReference\[oaicite:50]{index=50}

&nbsp;       self.depth\_pipe = hf\_pipeline(

&nbsp;           task="depth-estimation", 

&nbsp;           model="depth-anything/Depth-Anything-V2-Small-hf",

&nbsp;           device=0 if str(self.device) != "cpu" else -1

&nbsp;       )

&nbsp;       # (The pipeline will load the model weights on init. Ensure internet/HF cache is accessible for first load.)

&nbsp;   

&nbsp;   def prepare(self, \*\*kwargs) -> Requirements:

&nbsp;       """Specify how many input frames are needed before processing."""

&nbsp;       # We only need 1 frame at a time for depth estimation.

&nbsp;       return Requirements(input\_size=1)  # process frames one-by-one:contentReference\[oaicite:51]{index=51}

&nbsp;   

&nbsp;   def \_\_call\_\_(self, \*\*kwargs) -> dict:

&nbsp;       """Compute the depth map for the input frame.

&nbsp;       

&nbsp;       Args:

&nbsp;           video: List of input frame tensors (each of shape (1, H, W, C), values 0–255):contentReference\[oaicite:52]{index=52}.

&nbsp;           depth\_scale: (float) Scale factor for depth intensity adjustment.

&nbsp;       

&nbsp;       Returns:

&nbsp;           dict with "video": output frames tensor (shape (T, H, W, C), values in \[0,1]).

&nbsp;       """

&nbsp;       # Retrieve inputs

&nbsp;       video\_frames = kwargs.get("video")

&nbsp;       if video\_frames is None or len(video\_frames) == 0:

&nbsp;           raise ValueError("No video frame provided to DepthWebcamPipeline")

&nbsp;       frame\_tensor = video\_frames\[0]  # we requested 1 frame, so take the first

&nbsp;       # Convert torch tensor to PIL Image for the model

&nbsp;       # frame\_tensor shape: (1, H, W, C), values in \[0,255]

&nbsp;       frame\_np = frame\_tensor.squeeze(0).cpu().numpy().astype(np.uint8)  # shape (H, W, C)

&nbsp;       frame\_img = Image.fromarray(frame\_np)  # convert to PIL Image (RGB)

&nbsp;       

&nbsp;       # Run depth estimation model (Depth Anything V2) on the frame

&nbsp;       result = self.depth\_pipe(frame\_img)  # returns a dict with "predicted\_depth" or "depth"

&nbsp;       # HuggingFace depth pipeline returns a PIL Image in result\["depth"] by default:contentReference\[oaicite:53]{index=53}:contentReference\[oaicite:54]{index=54}

&nbsp;       depth\_img = result\["depth"]  # this is a PIL Image representing the depth map

&nbsp;       

&nbsp;       # Convert depth image to a torch tensor

&nbsp;       depth\_np = np.array(depth\_img)

&nbsp;       # If it's single-channel (H, W), add channel dimension

&nbsp;       if depth\_np.ndim == 2:

&nbsp;           depth\_np = depth\_np\[:, :, None]

&nbsp;       depth\_tensor = torch.from\_numpy(depth\_np).float()

&nbsp;       

&nbsp;       # Normalize depth values to \[0,1] range for output 

&nbsp;       # (DepthAnyV2 outputs relative depth; we'll scale per-frame for visualization)

&nbsp;       depth\_min = depth\_tensor.min()

&nbsp;       depth\_max = depth\_tensor.max()

&nbsp;       if depth\_max > depth\_min:

&nbsp;           depth\_tensor = (depth\_tensor - depth\_min) / (depth\_max - depth\_min)

&nbsp;       depth\_tensor = depth\_tensor.clamp(0.0, 1.0)

&nbsp;       

&nbsp;       # Apply the calibration scale from UI (depth\_scale) if provided

&nbsp;       # This can brighten/darken the depth map for calibration purposes

&nbsp;       depth\_scale = kwargs.get("depth\_scale", 1.0)

&nbsp;       depth\_tensor = (depth\_tensor \* depth\_scale).clamp(0.0, 1.0)

&nbsp;       

&nbsp;       # Ensure output is 3-channel RGB (repeat grayscale to RGB if needed)

&nbsp;       if depth\_tensor.size(-1) == 1:

&nbsp;           depth\_tensor = depth\_tensor.repeat(1, 1, 3)  # (H, W, 3)

&nbsp;       # Add time dimension T=1

&nbsp;       depth\_tensor = depth\_tensor.unsqueeze(0)  # shape (1, H, W, 3)

&nbsp;       

&nbsp;       return {"video": depth\_tensor}





Let’s break down the implementation:



Device and Model Initialization: In \_\_init\_\_, we set self.device to GPU if available. We then create a HuggingFace pipeline for depth estimation, specifying the Depth Anything V2 Small model. This will download/load the model weights (24.8M parameters) on first use. The model is loaded once and kept in memory for all frames. We pass device=0 for GPU or -1 for CPU as required (the code checks if device is not CPU) – this ensures the HF pipeline runs on the correct device. The chosen model "depth-anything/Depth-Anything-V2-Small-hf" is the small version of Depth Anything V2, which is Apache-2.0 and fastest. (Using the small model aligns with the “Any Webcam” requirement by being lightweight; larger models like V2-Base or Large could improve detail but would be slower and some are non-commercial licensed.)



Frame Requirements: The prepare() method tells Scope we need input\_size=1 – i.e., collect 1 frame then call our pipeline. This essentially means we process each frame individually. If we wanted to use multiple frames (e.g. for temporal filtering), we could request more, but for depth we don’t need that. Scope will pass a list of one frame to \_\_call\_\_ each time.



Input Frame Handling: In \_\_call\_\_, we get video\_frames from kwargs. Scope provides a list of frames equal to input\_size. We take the first frame tensor. This tensor has shape (1, H, W, C) where C=3 (RGB channels) and values range 0–255 (uint8). We convert it to a NumPy array and then to a PIL Image (frame\_img). We do this because the HF pipeline’s image processor expects a PIL image or NumPy array; using PIL ensures proper format (the image is interpreted in RGB). Note: If the color channels appear swapped, we may need to ensure correct mode (Scope likely provides frames in RGB order, given it’s using PyTorch tensors – we assume RGB here).



Depth Model Inference: We call self.depth\_pipe(frame\_img). The transformers depth pipeline returns a dictionary; according to HF docs, it may have keys like "predicted\_depth" (a tensor) and "depth" (a resized PIL image). In our use, we grab result\["depth"], which is a PIL Image representing the depth map output at the same resolution as input. (Internally, the model predicts at a lower resolution then upsamples to the original image size, so "depth" should be that upsampled result as an 8-bit image.) We then convert this PIL depth\_img to a NumPy array. At this point, depth\_np is essentially a grayscale image (shape H×W) where lighter/darker corresponds to different depths.



Normalization: We convert depth\_np to a torch tensor depth\_tensor. Depth values from the model are relative and not bounded to \[0,1]. They might be in arbitrary units or scales. For display purposes, we normalize each frame’s depth map to \[0,1] by min-max scaling. We compute depth\_min and depth\_max and scale accordingly. This ensures the nearest object in the frame becomes white (1.0) and the farthest becomes black (0.0), or vice versa, depending on how we interpret it. (If DepthAnything outputs inverse depth, nearer = larger values, then after normalization nearer parts will be white. Either way, the contrast is maximized per frame.) We clamp the result to \[0,1] as required by Scope (Scope expects output frames as float32 tensors in \[0,1] range for display).



Depth Scale Calibration: We retrieve the depth\_scale parameter from kwargs (this comes from the UI slider). By default it’s 1.0 (no scaling). We multiply the normalized depth by this factor. If depth\_scale > 1, the depth image brightens (highlights nearer differences more), if < 1, it darkens (which can help if some parts are saturating). This is a simple linear calibration. We clamp again to \[0,1] after scaling. Note that this doesn’t give physical units, but it allows the user to adjust the contrast of the depth visualization to match their environment or preference. In future, a more complex calibration could be implemented (e.g., mapping a known distance to a specific value, which would involve solving for scale and shift globally rather than per-frame normalization). If an actual metric depth model is used (one fine-tuned to predict in meters), this scale could be used to convert the output to meters if needed – but that would require knowing the model’s output unit. For now, we assume relative depth.



Output Formatting: We ensure the output tensor has 3 channels. If the depth map is grayscale (H×W×1), we repeat it into (H×W×3) so that Scope will treat it as an RGB video frame. This is done because Scope’s video rendering expects 3-channel images. (Alternatively, we could apply a colormap to the depth for a more colorful visualization, but here we keep it grayscale for simplicity – just duplicated in RGB channels). We then unsqueeze a time dimension to make the shape (1, H, W, 3), indicating a sequence of 1 frame (T=1). Finally, we return {"video": depth\_tensor}. Scope will take this and send it out as the next video frame in the stream.



This implementation will invert the colors such that closer objects are likely brighter (if using inverse depth, which DepthAnything V2 does by default). We could invert if needed by doing 1 - depth\_tensor before output to flip that convention (in some depth displays, nearer=white is common, but one can choose). The user can adjust via the slider if needed.



Importantly, we keep all heavy operations on the GPU: the HF pipeline will run the model on GPU (because we set device=0 if available), and the tensor operations for normalization are on CPU only briefly when converting from PIL, then moved to torch (we could optimize by avoiding the PIL round-trip using the AutoModel API directly, but the current approach is clear and leverages HF’s preprocessing). If performance needs improvement, one could initialize AutoModelForDepthEstimation and AutoImageProcessor outside the loop (as shown in HF docs) and then process the image tensor directly, avoiding PIL conversion every frame. But given DepthAnything V2 Small is quite fast and frames are reasonably sized (Scope might default to 512px or user-specified resolution), this pipeline should run near real-time on a GPU.



License note: The DepthAnything V2 Small model and code are Apache-2.0, so integrating them is permissible. If using a larger model (Base/Large) or certain fine-tuned weights, be mindful some are CC BY-NC (non-commercial) – we stick to the permissively licensed ones (Small or any others explicitly Apache). The HuggingFace model card confirms the small model’s license is Apache-2.0.



Deployment and Usage



Once the plugin is implemented and installed, here’s how to run and use it:



Launch Scope with GPU Access: If running locally, start Daydream Scope (desktop app or via CLI uv run daydream-scope). Ensure your machine has a capable GPU or, if not, consider running on a cloud GPU. If using a cloud provider like RunPod, use the daydreamlive/scope:main Docker image (Scope’s latest version) to start a container. This ensures plugin compatibility with the latest API. You might set up a remote desktop or port forwarding (Scope’s UI runs in Electron or a browser) to view the interface.



Install the Plugin: If not already installed, add the plugin. In the Scope app UI, go to Settings → Plugins, and enter the plugin’s Git URL (or local path) and click Install. Alternatively, use the CLI:



uv run daydream-scope install https://github.com/YourUser/scope-depth.git





Wait for Scope to download the plugin and restart. After restart, you should see no errors in the console – the plugin will appear in the Plugins list with its source.



Select the Depth Pipeline: In the Scope UI, there’s a pipeline selector (usually a dropdown at the top of the app). Switch to the pipeline named “Depth (Webcam)” (as we set in pipeline\_name) which corresponds to our depth-webcam pipeline. The app will load our pipeline (initializing the model, which may take a few seconds the first time as weights load).



Connect Webcam Feed: Ensure your webcam is plugged in and accessible. In the Scope interface, set the input source to your camera. Scope supports camera feeds as a source for video pipelines. If using the desktop app, it might default to the system’s default camera when a video pipeline is running. In case it doesn’t automatically start, look for an Input or Video Source control – for example, some versions have a toggle to switch between an uploaded video file and the live camera. Select your webcam (there may be a dropdown if multiple cameras are present). Once selected, you should see a preview from the camera. (If running headless or via tunnel, ensure you forward camera feed or test with a sample video file for verification.)



Start Streaming: Click the Play ▶️ button in Scope to begin streaming. The pipeline will now begin processing frames from the webcam. Within a moment, the output video canvas will show the depth map instead of the normal image. You should observe that the scene is in grayscale (or faint color) where nearer objects are lighter or darker relative to farther ones. Move objects or your hand in front of the camera – the depth output should update in real-time, indicating depth differences.



Adjust Settings: Open the Settings panel (gear icon) while the pipeline is running. Under our pipeline’s section (likely labeled "Depth Webcam Configuration"), you’ll find the “Depth Scale” slider. Try adjusting it: moving it above 1 will amplify depth contrasts (brightening the near/far differentiation), whereas below 1 will make the depth map more uniform. This can be used to calibrate the visualization if, say, everything looks too dark because the scene is very far away – increasing scale will spread out the depth values more. The change is applied live (it’s a runtime param, not requiring restart). You can reset it to 1.0 for no scaling.



Observation and Validation: Use some known arrangement to verify depth qualitatively – for example, place an object close to the camera and another far. The closer one should appear with a different intensity (e.g., brighter) than the far one, confirming the model’s relative depth ordering. Keep in mind that without metric calibration, the plugin does not output actual distances (e.g., it won’t tell you “3 meters” vs “5 meters” out-of-the-box). It’s relative depth – it excels at understanding what is in front of what. For many applications (like background separation, focus effects, or guiding 3D effects), relative depth is sufficient.



Metric Depth (Optional): If you require approximate real-world distances, there are two approaches: (a) Use a metric-trained model variant, or (b) implement a custom calibration procedure. Depth Anything V2 has versions fine-tuned for metric depth (e.g., on NYUv2 dataset) that attempt to predict in absolute units. For instance, the model "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf" exists for outdoor metric depth. We could incorporate a toggle in the config (e.g., use\_metric\_model=True as a load-time param) to load such a model. However, note that as of mid-2024, full support for metric output in Transformers was still catching up – one needed the latest transformer version to use those weights properly. If used, the metric model would output depth in some consistent scale (likely meters), but you’d need to verify and possibly adjust scaling. (b) Alternatively, a manual calibration could be done: for example, place a known-size object or measure the distance to a wall, then adjust a scale/offset until the depth map values correspond (this could be a future feature – e.g., clicking a point in the depth image and entering the real distance to compute a scale factor). Such calibration is not yet implemented here. For now, the plugin focuses on reliable relative depth.



Running on RunPod/Cloud: If you are running this in a cloud environment (because your local machine lacks a GPU), you’ll need to view the Scope UI remotely. One approach is to run Scope in a RunPod instance (using the main branch Docker as mentioned) and enable a secure tunnel (like ngrok or localhost.run) to access the interface. When using RunPod, open the “Enable SSH/HTTP” option to get a public URL for the Scope app, then open that in a browser. The plugin installation steps are the same. Keep in mind that the official Scope Cloud Inference (beta) does not support custom plugins, so you can’t use the Daydream cloud service for this yet – you must host the full Scope backend yourself on the cloud instance. Also, ensure camera access: if using RunPod and you want a live camera feed, it’s tricky (cloud servers don’t have a webcam). Instead, you might test with a pre-recorded video file on loop. Alternatively, run Scope on a local machine with a GPU or use something like an IP camera feed via a virtual device (beyond scope here).



Verification \& Future Work: Confirm that the depth plugin runs steadily (watch for any dropped frames or high latency – DepthAnything V2 Small should run at several frames per second on a mid-range GPU, e.g., RTX A4000 as recommended, and even realtime ~30fps on high-end GPUs). If performance is an issue, consider reducing input resolution or using the model’s smaller variants (there’s an even smaller ViT-S model, or run the frames at 384px instead of full resolution). For future enhancements, one can implement temporal smoothing to make depth maps more temporally consistent (DepthAnything V2 by itself is fairly consistent, but slight flicker can occur frame-to-frame). Also, integration with Scope’s VACE system could allow using our depth maps as control signals for generative pipelines (Scope’s built-in video-depth-anything preprocessor already demonstrates this). Since our plugin is a full pipeline, it’s primarily for visualization or further custom use (perhaps feeding into a custom 3D effect). If the goal is to combine the depth with another pipeline (e.g., apply an AR effect to the webcam based on depth), one could refactor this into a preprocessor plugin once Scope exposes that interface to plugins.



In summary, this specification covers all components to build a Depth Anything V2 webcam plugin: plugin configuration, model integration with permissive licensed code, and deployment details. By following the above steps and code, a coding agent can implement the plugin and achieve real-time monocular depth estimation on a webcam feed within Daydream Scope. The combination of Scope’s plugin architecture and DepthAnything V2’s accuracy provides a powerful tool for depth-based AI video

