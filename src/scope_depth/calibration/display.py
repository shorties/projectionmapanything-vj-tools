"""Fullscreen calibration pattern display for projector output."""

import tkinter as tk
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


class CalibrationDisplay:
    """Handles fullscreen display of calibration patterns on projector.

    Uses tkinter for cross-platform fullscreen window management.
    Supports selecting specific monitor for multi-display setups.
    """

    def __init__(self):
        """Initialize display handler."""
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.current_image: Optional[np.ndarray] = None
        self.is_fullscreen = False
        self.monitor_index = 0

        self.key_callback: Optional[Callable[[str], None]] = None

    def get_available_monitors(self) -> List[Tuple[int, str, int, int]]:
        """Get list of available monitors.

        Returns:
            List of (index, name, width, height) tuples
        """
        temp_root = tk.Tk()
        temp_root.withdraw()

        monitors = []

        primary_w = temp_root.winfo_screenwidth()
        primary_h = temp_root.winfo_screenheight()
        monitors.append((0, f"Primary ({primary_w}x{primary_h})", primary_w, primary_h))

        try:
            temp_root.update_idletasks()
            virtual_w = temp_root.winfo_vrootwidth()
            virtual_h = temp_root.winfo_vrootheight()

            if virtual_w > primary_w or virtual_h > primary_h:
                monitors.append((
                    1,
                    f"Secondary ({virtual_w - primary_w}x{primary_h})",
                    virtual_w - primary_w,
                    primary_h,
                ))
        except Exception:
            pass

        temp_root.destroy()
        return monitors

    def start_fullscreen(
        self,
        monitor_index: int = 0,
        on_key: Optional[Callable[[str], None]] = None,
    ) -> Tuple[int, int]:
        """Start fullscreen display on selected monitor.

        Args:
            monitor_index: Index of monitor to use (0 = primary)
            on_key: Callback for key press events

        Returns:
            (width, height) of the selected monitor
        """
        self.monitor_index = monitor_index
        self.key_callback = on_key

        self.root = tk.Tk()
        self.root.withdraw()

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        x_offset = 0
        if monitor_index > 0:
            x_offset = screen_w

        self.root.overrideredirect(True)
        self.root.geometry(f"{screen_w}x{screen_h}+{x_offset}+0")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="black")

        self.canvas = tk.Canvas(
            self.root,
            width=screen_w,
            height=screen_h,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<Key>", self._on_key_press)
        self.root.bind("<Escape>", lambda e: self.stop())
        self.root.bind("<Button-1>", lambda e: self.stop())

        self.root.deiconify()
        self.root.update()
        self.is_fullscreen = True

        return screen_w, screen_h

    def show_image(self, image: np.ndarray) -> None:
        """Display an image on the fullscreen canvas.

        Args:
            image: BGR image array (H, W, 3) from OpenCV
        """
        if self.root is None or self.canvas is None:
            return

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.current_image = image

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            image=self.photo_image,
            anchor=tk.CENTER,
        )

        self.root.update_idletasks()
        self.root.update()

    def show_text(self, text: str, size: int = 48) -> None:
        """Display text on black background.

        Args:
            text: Text to display
            size: Font size
        """
        if self.root is None:
            return

        canvas_w = self.canvas.winfo_width() if self.canvas else 1920
        canvas_h = self.canvas.winfo_height() if self.canvas else 1080

        img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size_cv = cv2.getTextSize(text, font, size / 30, 2)[0]
        text_x = (canvas_w - text_size_cv[0]) // 2
        text_y = (canvas_h + text_size_cv[1]) // 2

        # Support multi-line text
        lines = text.split("\n")
        line_height = text_size_cv[1] + 20
        start_y = text_y - (len(lines) - 1) * line_height // 2

        for i, line in enumerate(lines):
            line_size = cv2.getTextSize(line, font, size / 30, 2)[0]
            line_x = (canvas_w - line_size[0]) // 2
            line_y = start_y + i * line_height
            cv2.putText(img, line, (line_x, line_y), font, size / 30, (255, 255, 255), 2)

        self.show_image(img)

    def show_progress(self, current: int, total: int, label: str = "Pattern") -> None:
        """Show progress bar.

        Args:
            current: Current step (1-based)
            total: Total steps
            label: Label to display
        """
        if self.root is None:
            return

        canvas_w = self.canvas.winfo_width() if self.canvas else 1920
        canvas_h = self.canvas.winfo_height() if self.canvas else 1080

        img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Draw progress bar
        bar_w = canvas_w * 0.6
        bar_h = 40
        bar_x = (canvas_w - bar_w) // 2
        bar_y = canvas_h // 2

        # Background
        cv2.rectangle(
            img,
            (int(bar_x), int(bar_y - bar_h // 2)),
            (int(bar_x + bar_w), int(bar_y + bar_h // 2)),
            (50, 50, 50),
            -1,
        )

        # Progress
        progress = current / total
        cv2.rectangle(
            img,
            (int(bar_x), int(bar_y - bar_h // 2)),
            (int(bar_x + bar_w * progress), int(bar_y + bar_h // 2)),
            (0, 255, 0),
            -1,
        )

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{label} {current}/{total}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (canvas_w - text_size[0]) // 2
        text_y = int(bar_y - bar_h)

        cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

        self.show_image(img)

    def update(self) -> None:
        """Process pending GUI events."""
        if self.root:
            self.root.update_idletasks()
            self.root.update()

    def stop(self) -> None:
        """Stop fullscreen display."""
        if self.root:
            self.root.destroy()
            self.root = None
        self.canvas = None
        self.photo_image = None
        self.is_fullscreen = False

    def _on_key_press(self, event: tk.Event) -> None:
        """Handle key press."""
        if self.key_callback and event.char:
            self.key_callback(event.char)

    def is_running(self) -> bool:
        """Check if display is active."""
        return self.root is not None and self.is_fullscreen


class StructuredLightSequence:
    """Manages structured light calibration sequence."""

    def __init__(
        self,
        gray_encoder,
        display: CalibrationDisplay,
    ):
        """Initialize structured light sequence.

        Args:
            gray_encoder: GrayCodeEncoder instance
            display: CalibrationDisplay instance
        """
        self.gray_encoder = gray_encoder
        self.display = display
        self.patterns = gray_encoder.generate_patterns()
        self.current_index = 0
        self.captured_images: List[np.ndarray] = []
        self.is_running = False

        self.on_pattern_change: Optional[Callable[[int, int], None]] = None
        self.on_complete: Optional[Callable[[List[np.ndarray]], None]] = None

    def start(
        self,
        on_pattern_change: Optional[Callable[[int, int], None]] = None,
        on_complete: Optional[Callable[[List[np.ndarray]], None]] = None,
    ) -> None:
        """Start structured light sequence."""
        self.on_pattern_change = on_pattern_change
        self.on_complete = on_complete
        self.current_index = 0
        self.captured_images = []
        self.is_running = True

        self._show_current_pattern()

    def _show_current_pattern(self) -> None:
        """Display current pattern."""
        if not self.is_running or self.current_index >= len(self.patterns):
            self._finish()
            return

        pattern = self.patterns[self.current_index]
        self.display.show_image(pattern)

        if self.on_pattern_change:
            self.on_pattern_change(self.current_index + 1, len(self.patterns))

    def capture_frame(self, camera_frame: np.ndarray) -> bool:
        """Capture current pattern."""
        if not self.is_running:
            return False

        self.captured_images.append(camera_frame.copy())
        self.current_index += 1
        self._show_current_pattern()
        return True

    def _finish(self) -> None:
        """Complete sequence."""
        self.is_running = False

        # Show white for capture validation
        white = np.ones(self.patterns[0].shape, dtype=np.uint8) * 255
        self.display.show_image(white)

        if self.on_complete:
            self.on_complete(self.captured_images)

    def cancel(self) -> None:
        """Cancel sequence."""
        self.is_running = False
        self.captured_images = []


class CheckerboardSequence:
    """Manages checkerboard camera calibration sequence."""

    def __init__(
        self,
        calibrator,
        display: CalibrationDisplay,
        num_positions: int = 15,
    ):
        """Initialize checkerboard sequence.

        Args:
            calibrator: ProCamCalibrator instance
            display: CalibrationDisplay
            num_positions: Number of checkerboard positions to capture
        """
        self.calibrator = calibrator
        self.display = display
        self.num_positions = num_positions
        self.captured_images: List[np.ndarray] = []
        self.is_running = False
        self.current_position = 0

        self.on_capture: Optional[Callable[[int, int], None]] = None
        self.on_complete: Optional[Callable[[List[np.ndarray]], None]] = None

    def start(
        self,
        on_capture: Optional[Callable[[int, int], None]] = None,
        on_complete: Optional[Callable[[List[np.ndarray]], None]] = None,
    ) -> None:
        """Start checkerboard capture sequence."""
        self.on_capture = on_capture
        self.on_complete = on_complete
        self.current_position = 0
        self.captured_images = []
        self.is_running = True

        self._show_instruction()

    def _show_instruction(self) -> None:
        """Show capture instruction."""
        if not self.is_running:
            return

        self.display.show_text(
            f"Camera Calibration {self.current_position + 1}/{self.num_positions}\n"
            "Show checkerboard to camera\n"
            "Press SPACE to capture or S to skip"
        )

    def attempt_capture(self, camera_frame: np.ndarray) -> bool:
        """Try to capture checkerboard from frame.

        Returns:
            True if checkerboard was detected and captured
        """
        if not self.is_running:
            return False

        # Check if checkerboard is visible
        corners = self.calibrator.calibrator.detect_checkerboard(camera_frame)

        if corners is not None:
            self.captured_images.append(camera_frame.copy())
            self.current_position += 1

            if self.on_capture:
                self.on_capture(self.current_position, self.num_positions)

            if self.current_position >= self.num_positions:
                self._finish()
            else:
                self._show_instruction()

            return True

        return False

    def skip(self) -> None:
        """Skip current position."""
        self.current_position += 1
        if self.current_position >= self.num_positions:
            self._finish()
        else:
            self._show_instruction()

    def _finish(self) -> None:
        """Complete sequence."""
        self.is_running = False
        self.display.show_text("Camera calibration complete!")

        if self.on_complete:
            self.on_complete(self.captured_images)

    def cancel(self) -> None:
        """Cancel sequence."""
        self.is_running = False
        self.captured_images = []
