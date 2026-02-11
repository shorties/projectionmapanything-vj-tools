"""Projector-camera calibration using structured light and stereo calibration.

This implements a ProCam calibration approach similar to RoomAlive Toolkit:
1. Camera intrinsics calibration using checkerboard
2. Structured light (Gray code) for dense projector-camera correspondences
3. Stereo calibration to find camera-projector relative pose
4. Full distortion model for both camera and projector
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    K: np.ndarray  # 3x3 intrinsic matrix
    dist_coeffs: np.ndarray  # distortion coefficients (k1, k2, p1, p2, k3)
    image_size: Tuple[int, int]  # (width, height)
    reprojection_error: float = 0.0


@dataclass
class ProCamCalibration:
    """Complete projector-camera calibration result."""

    # Camera intrinsics
    camera: CameraIntrinsics

    # Projector "intrinsics" (treated as second camera)
    projector: CameraIntrinsics

    # Stereo extrinsics (camera -> projector transform)
    R: np.ndarray  # 3x3 rotation matrix
    T: np.ndarray  # 3x1 translation vector
    E: np.ndarray  # 3x3 essential matrix
    F: np.ndarray  # 3x3 fundamental matrix

    # Validation metrics
    stereo_error: float = 0.0
    is_valid: bool = False

    def project_points_to_projector(
        self,
        points_3d: np.ndarray,
    ) -> np.ndarray:
        """Project 3D points to projector image plane.

        Args:
            points_3d: Nx3 array of 3D points in camera coordinate system

        Returns:
            Nx2 array of projected points in projector image coordinates
        """
        # Transform points from camera to projector coordinate system
        points_proj = (self.R @ points_3d.T + self.T).T

        # Project to projector image plane
        points_2d, _ = cv2.projectPoints(
            points_proj,
            np.zeros(3),
            np.zeros(3),
            self.projector.K,
            self.projector.dist_coeffs,
        )

        return points_2d.reshape(-1, 2)

    def warp_camera_to_projector(
        self,
        image: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Warp camera image to projector perspective using stereo rectification.

        Args:
            image: Input camera image
            output_size: Desired output size (defaults to projector size)

        Returns:
            Warped image in projector perspective
        """
        if output_size is None:
            output_size = self.projector.image_size

        # Compute rectification maps
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.camera.K,
            self.camera.dist_coeffs,
            self.projector.K,
            self.projector.dist_coeffs,
            self.camera.image_size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,  # Crop to valid pixels
        )

        # Compute undistort-rectify maps for camera
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera.K,
            self.camera.dist_coeffs,
            R1,
            P1,
            self.camera.image_size,
            cv2.CV_32FC1,
        )

        # Rectify camera image
        rectified = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

        # Now we need to map from rectified camera to projector
        # For this, we compute a homography from rectified camera to projector
        # using the stereo geometry

        # Create grid of points in rectified camera image
        h, w = self.camera.image_size[1], self.camera.image_size[0]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        camera_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

        # Triangulate to 3D (simplified - assumes roughly planar or uses depth)
        # For proper mapping, we use the fundamental matrix relationship
        # F * x_camera * x_projector = 0

        # Alternative: compute dense optical flow or use disparity
        # For now, use a simplified homography approach with rectification

        # Compute homography from rectified camera to projector
        # Sample correspondences at grid points
        sample_pts_cam = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
            [w // 2, h // 2],
            [w // 4, h // 4],
            [3 * w // 4, h // 4],
            [3 * w // 4, 3 * h // 4],
            [w // 4, 3 * h // 4],
        ], dtype=np.float32)

        # Project these to projector
        # First undistort and rectify camera points
        pts_undistorted = cv2.undistortPoints(
            sample_pts_cam.reshape(-1, 1, 2),
            self.camera.K,
            self.camera.dist_coeffs,
            R=R1,
            P=P1,
        )

        # Triangulate to get 3D (at unit depth for homography estimation)
        pts_3d = np.hstack([pts_undistorted.reshape(-1, 2), np.ones((len(pts_undistorted), 1))])

        # Transform to projector coords and project
        pts_proj = self.project_points_to_projector(pts_3d)

        # Compute homography
        H, _ = cv2.findHomography(sample_pts_cam, pts_proj, cv2.RANSAC, 5.0)

        if H is not None:
            warped = cv2.warpPerspective(rectified, H, output_size)
            return warped

        return rectified

    def undistort_camera_image(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from camera image.

        Args:
            image: Distorted camera image

        Returns:
            Undistorted image
        """
        return cv2.undistort(
            image,
            self.camera.K,
            self.camera.dist_coeffs,
        )

    def get_optimal_new_camera_matrix(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Get optimal camera matrix for undistortion with minimal cropping.

        Returns:
            (new_camera_matrix, roi)
        """
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.camera.K,
            self.camera.dist_coeffs,
            self.camera.image_size,
            1,  # alpha
            self.camera.image_size,
        )
        return new_K, roi


class GrayCodeEncoder:
    """Gray code structured light encoder/decoder.

    Gray codes are used to establish dense correspondences between
    camera and projector by projecting binary patterns.
    """

    def __init__(self, width: int, height: int):
        """Initialize Gray code encoder.

        Args:
            width: Projector width in pixels
            height: Projector height in pixels
        """
        self.width = width
        self.height = height
        self.num_horizontal_bits = int(np.ceil(np.log2(width)))
        self.num_vertical_bits = int(np.ceil(np.log2(height)))

    def generate_patterns(self) -> List[np.ndarray]:
        """Generate all Gray code patterns.

        Returns:
            List of BGR images (alternating horizontal and vertical patterns)
            Pattern order: H0, H0_inv, H1, H1_inv, ..., V0, V0_inv, V1, V1_inv, ...
        """
        patterns = []

        # Horizontal patterns (encode column positions)
        for bit in range(self.num_horizontal_bits):
            pattern = self._create_gray_code_pattern(
                self.width,
                self.height,
                bit,
                horizontal=True,
            )
            patterns.append(pattern)
            patterns.append(cv2.bitwise_not(pattern))  # Inverse

        # Vertical patterns (encode row positions)
        for bit in range(self.num_vertical_bits):
            pattern = self._create_gray_code_pattern(
                self.width,
                self.height,
                bit,
                horizontal=False,
            )
            patterns.append(pattern)
            patterns.append(cv2.bitwise_not(pattern))  # Inverse

        # Add all-white and all-black for normalization
        white = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        patterns.append(white)
        patterns.append(black)

        return patterns

    def _create_gray_code_pattern(
        self,
        width: int,
        height: int,
        bit: int,
        horizontal: bool,
    ) -> np.ndarray:
        """Create a single Gray code pattern.

        Args:
            width: Pattern width
            height: Pattern height
            bit: Which bit position (0 = LSB)
            horizontal: True for column encoding, False for row encoding

        Returns:
            BGR pattern image
        """
        pattern = np.zeros((height, width), dtype=np.uint8)

        if horizontal:
            # Encode column positions
            for x in range(width):
                # Convert to Gray code
                gray = x ^ (x >> 1)
                # Check if bit is set
                if (gray >> bit) & 1:
                    pattern[:, x] = 255
        else:
            # Encode row positions
            for y in range(height):
                gray = y ^ (y >> 1)
                if (gray >> bit) & 1:
                    pattern[y, :] = 255

        return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

    def decode_correspondences(
        self,
        captured_patterns: List[np.ndarray],
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode Gray code patterns to find camera-projector correspondences.

        Args:
            captured_patterns: List of captured camera images (in same order as projected)
            threshold: Binary threshold (0-1) for pattern decoding

        Returns:
            (camera_points, projector_points) arrays of corresponding 2D coordinates
        """
        num_h_patterns = self.num_horizontal_bits * 2  # pattern + inverse
        num_v_patterns = self.num_vertical_bits * 2

        h_patterns = captured_patterns[:num_h_patterns]
        v_patterns = captured_patterns[num_h_patterns:num_h_patterns + num_v_patterns]
        white_img = captured_patterns[-2]
        black_img = captured_patterns[-1]

        # Normalize images
        white_gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        black_gray = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = white_gray - black_gray + 1e-6

        # Decode horizontal bits for each pixel
        h, w = white_gray.shape
        proj_x = np.zeros((h, w), dtype=np.int32)

        for bit in range(self.num_horizontal_bits):
            pos_idx = bit * 2
            neg_idx = bit * 2 + 1

            pos_img = cv2.cvtColor(h_patterns[pos_idx], cv2.COLOR_BGR2GRAY).astype(np.float32)
            neg_img = cv2.cvtColor(h_patterns[neg_idx], cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Decode bit using difference from white/black
            bit_value = ((pos_img - neg_img) / diff) > threshold
            proj_x |= (bit_value.astype(np.int32) << bit)

        # Convert Gray code to binary
        proj_x = self._gray_to_binary(proj_x)

        # Decode vertical bits
        proj_y = np.zeros((h, w), dtype=np.int32)

        for bit in range(self.num_vertical_bits):
            pos_idx = bit * 2
            neg_idx = bit * 2 + 1

            pos_img = cv2.cvtColor(v_patterns[pos_idx], cv2.COLOR_BGR2GRAY).astype(np.float32)
            neg_img = cv2.cvtColor(v_patterns[neg_idx], cv2.COLOR_BGR2GRAY).astype(np.float32)

            bit_value = ((pos_img - neg_img) / diff) > threshold
            proj_y |= (bit_value.astype(np.int32) << bit)

        proj_y = self._gray_to_binary(proj_y)

        # Create correspondence arrays
        # Valid correspondences are where decoded values are within projector bounds
        valid_mask = (proj_x >= 0) & (proj_x < self.width) & (proj_y >= 0) & (proj_y < self.height)

        # Also require sufficient contrast
        contrast_mask = diff > 20
        valid_mask &= contrast_mask

        # Get camera coordinates (pixel positions in captured image)
        cam_y, cam_x = np.where(valid_mask)
        camera_points = np.stack([cam_x, cam_y], axis=1).astype(np.float32)
        projector_points = np.stack([proj_x[valid_mask], proj_y[valid_mask]], axis=1).astype(np.float32)

        return camera_points, projector_points

    @staticmethod
    def _gray_to_binary(gray_code: np.ndarray) -> np.ndarray:
        """Convert Gray code to binary.

        Args:
            gray_code: Array of Gray code values

        Returns:
            Array of binary values
        """
        binary = gray_code.copy()
        shift = 1
        while (gray_code >> shift).any():
            binary ^= (gray_code >> shift)
            shift += 1
        return binary


class ProCamCalibrator:
    """Main calibrator for camera-projector systems.

    Implements calibration workflow:
    1. Calibrate camera intrinsics using checkerboard
    2. Use structured light for dense correspondences
    3. Stereo calibration for full camera-projector geometry
    """

    # Checkerboard defaults for camera calibration
    CHECKERBOARD_ROWS = 6
    CHECKERBOARD_COLS = 9
    SQUARE_SIZE = 25.0  # mm, for scale (optional)

    def __init__(
        self,
        projector_size: Tuple[int, int] = (1920, 1080),
        checkerboard_rows: int = CHECKERBOARD_ROWS,
        checkerboard_cols: int = CHECKERBOARD_COLS,
    ):
        """Initialize calibrator.

        Args:
            projector_size: (width, height) of projector
            checkerboard_rows: Rows in checkerboard pattern
            checkerboard_cols: Columns in checkerboard pattern
        """
        self.projector_size = projector_size
        self.checkerboard_size = (checkerboard_cols, checkerboard_rows)
        self.square_size = self.SQUARE_SIZE

        # Gray code encoder for structured light
        self.gray_encoder = GrayCodeEncoder(projector_size[0], projector_size[1])

        # Camera calibration data
        self.camera_object_points: List[np.ndarray] = []
        self.camera_image_points: List[np.ndarray] = []

        # Stereo calibration data
        self.stereo_camera_points: List[np.ndarray] = []
        self.stereo_projector_points: List[np.ndarray] = []

        # Results
        self.camera_intrinsics: Optional[CameraIntrinsics] = None
        self.procams: List[ProCamCalibration] = []

    def calibrate_camera(
        self,
        images: List[np.ndarray],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> CameraIntrinsics:
        """Calibrate camera intrinsics using checkerboard patterns.

        Args:
            images: List of captured checkerboard images (camera perspective)
            image_size: (width, height) if known, else auto-detected

        Returns:
            CameraIntrinsics with K, distortion coefficients
        """
        objp = np.zeros(
            (self.checkerboard_size[0] * self.checkerboard_size[1], 3),
            np.float32,
        )
        objp[:, :2] = (
            np.mgrid[0 : self.checkerboard_size[0], 0 : self.checkerboard_size[1]].T.reshape(-1, 2)
            * self.square_size
        )

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if image_size is None:
                h, w = gray.shape[:2]
                image_size = (w, h)

            ret, corners = cv2.findChessboardCorners(
                gray,
                self.checkerboard_size,
                None,
            )

            if ret:
                self.camera_object_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.camera_image_points.append(corners2)

        if len(self.camera_object_points) < 3:
            raise ValueError(
                f"Need at least 3 checkerboard images, found {len(self.camera_object_points)}"
            )

        # Calibrate camera
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.camera_object_points,
            self.camera_image_points,
            image_size,
            None,
            None,
        )

        if not ret:
            raise RuntimeError("Camera calibration failed")

        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.camera_object_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.camera_object_points[i],
                rvecs[i],
                tvecs[i],
                K,
                dist,
            )
            error = cv2.norm(self.camera_image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(self.camera_object_points)

        self.camera_intrinsics = CameraIntrinsics(
            K=K,
            dist_coeffs=dist.flatten(),
            image_size=image_size,
            reprojection_error=mean_error,
        )

        return self.camera_intrinsics

    def add_structured_light_correspondences(
        self,
        captured_patterns: List[np.ndarray],
    ) -> int:
        """Add correspondences from structured light patterns.

        Args:
            captured_patterns: List of images captured for each projected pattern

        Returns:
            Number of correspondences found
        """
        camera_pts, projector_pts = self.gray_encoder.decode_correspondences(
            captured_patterns
        )

        if len(camera_pts) < 100:
            raise ValueError(
                f"Too few correspondences found: {len(camera_pts)} (need at least 100)"
            )

        # Subsample if too many points (for performance)
        if len(camera_pts) > 5000:
            indices = np.random.choice(len(camera_pts), 5000, replace=False)
            camera_pts = camera_pts[indices]
            projector_pts = projector_pts[indices]

        self.stereo_camera_points.append(camera_pts)
        self.stereo_projector_points.append(projector_pts)

        return len(camera_pts)

    def calibrate_stereo(self) -> ProCamCalibration:
        """Perform stereo calibration between camera and projector.

        Requires:
            - Camera intrinsics already calibrated
            - At least one set of stereo correspondences added

        Returns:
            ProCamCalibration with full camera-projector geometry
        """
        if self.camera_intrinsics is None:
            raise RuntimeError("Must calibrate camera intrinsics first")

        if len(self.stereo_camera_points) < 1:
            raise RuntimeError("Need at least one set of stereo correspondences")

        # Concatenate all stereo points
        all_camera_pts = np.vstack(self.stereo_camera_points)
        all_projector_pts = np.vstack(self.stereo_projector_points)

        # For stereo calibration, we need 3D object points
        # Since projector doesn't directly observe 3D, we triangulate from camera
        # A simplified approach: assume correspondences are at various depths
        # and use cv2.stereoCalibrate with computed object points

        # Alternative: Use cv2.calibrateCamera for "projector" treating projector
        # points as if they were imaged by a second camera
        # Then compute stereo extrinsics

        # First, calibrate "projector" as a camera
        # We need object points in 3D space - we'll triangulate from camera
        # For now, use a simplified approach with homography decomposition

        # Compute homography between camera and projector image planes
        H, mask = cv2.findHomography(all_camera_pts, all_projector_pts, cv2.RANSAC, 5.0)

        if H is None:
            raise RuntimeError("Could not compute camera-projector homography")

        # Decompose homography to get initial R, T estimates
        # This is approximate but gives us a starting point
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(
            H,
            self.camera_intrinsics.K,
        )

        # Select best solution (positive depth assumption)
        best_R, best_T = None, None
        for i in range(num_solutions):
            # Check if translation points towards scene (positive Z)
            if translations[i][2] > 0:
                best_R = rotations[i]
                best_T = translations[i]
                break

        if best_R is None:
            best_R = rotations[0]
            best_T = translations[0]

        # Refine using stereoCalibrate
        # Create object points by back-projecting camera points to unit depth
        # and transforming to common coordinate system
        camera_pts_norm = cv2.undistortPoints(
            all_camera_pts[:1000].reshape(-1, 1, 2),  # Limit for performance
            self.camera_intrinsics.K,
            self.camera_intrinsics.dist_coeffs,
        )

        # Use as object points at Z=1
        object_points = np.hstack([
            camera_pts_norm.reshape(-1, 2),
            np.ones((len(camera_pts_norm), 1))
        ])

        # Projector points for these object points (should match if R, T correct)
        projector_pts_subset = all_projector_pts[:1000]

        # Calibrate projector as second camera
        ret, K_proj, dist_proj, rvecs, tvecs = cv2.calibrateCamera(
            [object_points.astype(np.float32)],
            [projector_pts_subset.reshape(-1, 1, 2)],
            self.projector_size,
            None,
            None,
            flags=cv2.CALIB_FIX_PRINCIPAL_POINT
            + cv2.CALIB_FIX_ASPECT_RATIO
            + cv2.CALIB_ZERO_TANGENT_DIST,
        )

        if not ret:
            raise RuntimeError("Projector calibration failed")

        # Now stereo calibration
        # Reformat points for stereoCalibrate
        object_points_list = [object_points.astype(np.float32)]
        camera_points_list = [
            all_camera_pts[:1000].reshape(-1, 1, 2).astype(np.float32)
        ]
        projector_points_list = [
            all_projector_pts[:1000].reshape(-1, 1, 2).astype(np.float32)
        ]

        # Use initial guess from homography decomposition
        R_init = best_R
        T_init = best_T.flatten()

        flags = (
            cv2.CALIB_FIX_INTRINSIC
            + cv2.CALIB_USE_EXTRINSIC_GUESS
            + cv2.CALIB_FIX_ASPECT_RATIO
        )

        ret, K_cam, dist_cam, K_proj, dist_proj, R, T, E, F = cv2.stereoCalibrate(
            object_points_list,
            camera_points_list,
            projector_points_list,
            self.camera_intrinsics.K,
            self.camera_intrinsics.dist_coeffs,
            K_proj,
            dist_proj.flatten(),
            self.camera_intrinsics.image_size,
            R_init,
            T_init,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        )

        if not ret:
            raise RuntimeError("Stereo calibration failed")

        # Calculate stereo error
        # Project camera points through stereo geometry and compare to projector points
        total_error = 0.0
        for i in range(len(camera_points_list)):
            # Undistort camera points
            pts_undistorted = cv2.undistortPoints(
                camera_points_list[i],
                K_cam,
                dist_cam,
            )

            # Triangulate (simplified)
            pts_3d = cv2.convertPointsToHomogeneous(pts_undistorted).reshape(-1, 4)
            pts_3d[:, :3] *= 1000  # Scale to mm

            # Transform to projector coords
            pts_proj = (R @ pts_3d[:, :3].T).T + T.flatten()

            # Project to projector image
            pts_proj_2d, _ = cv2.projectPoints(
                pts_proj,
                np.zeros(3),
                np.zeros(3),
                K_proj,
                dist_proj.flatten(),
            )

            error = cv2.norm(projector_points_list[i], pts_proj_2d, cv2.NORM_L2)
            total_error += error / len(pts_proj_2d)

        stereo_error = total_error / len(camera_points_list)

        # Create calibration result
        camera = CameraIntrinsics(
            K=K_cam,
            dist_coeffs=dist_cam.flatten(),
            image_size=self.camera_intrinsics.image_size,
            reprojection_error=self.camera_intrinsics.reprojection_error,
        )

        projector = CameraIntrinsics(
            K=K_proj,
            dist_coeffs=dist_proj.flatten(),
            image_size=self.projector_size,
            reprojection_error=0.0,  # Would need separate validation
        )

        procam = ProCamCalibration(
            camera=camera,
            projector=projector,
            R=R,
            T=T.flatten(),
            E=E,
            F=F,
            stereo_error=stereo_error,
            is_valid=stereo_error < 5.0,  # pixels
        )

        self.procams.append(procam)
        return procam

    def clear(self) -> None:
        """Clear all calibration data."""
        self.camera_object_points = []
        self.camera_image_points = []
        self.stereo_camera_points = []
        self.stereo_projector_points = []
        self.camera_intrinsics = None
        self.procams = []
