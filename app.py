import cv2
import numpy as np
import time
import os
import cv2.aruco

def draw_text(frame, text, pos=(20, 50), scale=1.0, color=(255, 255, 255)):
    """Draws white text with a black outline for better visibility."""
    thickness = int(scale * 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

class LiveCameraEffects:
    def __init__(self):
        # Camera Init
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.window_name = "Live Camera Effects"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # State variables
        self.mode = "raw"
        self.contrast = 1.0
        self.brightness = 0
        self.rotation_angle = 0
        
        # Filter parameters
        self.gaussian_kernel_size = 9
        self.bilateral_d = 9

        # Panorama state
        self.panorama_base = None
        self.panorama_frames = []
        self.stitch_status_message = ""
        self.stitch_status_message_time = 0


        # Calibration/Undistort State
        self.mtx = None
        self.dist = None
        self.calibration_results = {} # To hold results for on-screen display
        self.CHESSBOARD_SIZE = (9, 6)
        self.SQUARE_SIZE_MM = 25
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.CHESSBOARD_SIZE[0], 0:self.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.objp = self.objp * self.SQUARE_SIZE_MM
        self.objpoints, self.imgpoints = [], []
        self.images_captured, self.TARGET_IMAGES = 0, 20
        self.last_capture_time = time.time()
        
        # --- AR State ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self._load_calibration() # Loads for both AR and Undistort

        # --- 3D Model State ---
        self.model_vertices = []
        self.model_faces = []
        self._load_obj('trex_model.obj')

        # Controls list
        self.help_text = [
            "[q] Quit",
            "[0] Raw   [1] Gray   [2] HSV",
            "[a] Augmented Reality",
            "[c/v] Contrast +/-   [b/n] Brightness +/-",
            "[g] Gaussian Blur   [f] Bilateral Filter",
            "    - G_Kernel: [ ]    B_Diameter: < >",
            "[e] Canny   [l] Hough Lines   [h] Histogram",
            "[t] Translate   [r] Rotate   [s] Scale",
            "[k] Calibrate   [u] Undistort View",
            "[o] Capture Frame   [p] Stitch Panorama   [z] Reset"
        ]

    def _load_calibration(self):
        """Loads camera calibration data from a file."""
        if os.path.exists('calibration.npz'):
            with np.load('calibration.npz') as X:
                self.mtx, self.dist = X['mtx'], X['dist']
            print("Calibration data loaded.")
        else:
            print("WARNING: 'calibration.npz' not found. AR and Undistort modes will not be accurate.")

    def _load_obj(self, filename, scale_factor=0.0003):
        """Loads a .OBJ file, scales and centers the model."""
        if not os.path.exists(filename):
            print(f"ERROR: Model file '{filename}' not found.")
            return

        vertices = []
        faces = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.split()
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('f '):
                        parts = line.split()[1:]
                        face = [int(p.split('/')[0]) - 1 for p in parts]
                        faces.append(face)

            if not vertices:
                print("ERROR: No vertices found in the OBJ file.")
                return

            vertices_np = np.array(vertices, dtype=np.float32)
            center = np.mean(vertices_np, axis=0)
            centered_vertices = vertices_np - center
            self.model_vertices = centered_vertices * scale_factor
            self.model_faces = faces
            print(f"Successfully loaded and processed model '{filename}'.")

        except Exception as e:
            print(f"Error loading OBJ file {filename}: {e}")

    def _calculate_homography(self, img1, img2):
        """Finds the homography matrix using SIFT and FLANN."""
        # Use SIFT to detect keypoints and descriptors.
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return None

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

        # Apply Lowe's ratio test to filter good matches.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography matrix.
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            return H
        return None

    def _stitch_images(self, img_left, img_right):
        """Stitches the right image to the left image."""
        H = self._calculate_homography(img_left, img_right)
        if H is None:
            print("Homography could not be computed.")
            return None

        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        # Warp the right image to align with the left image.
        result = cv2.warpPerspective(img_right, H, (w_left + w_right, max(h_left, h_right)))
        
        # Place the left image onto the result canvas.
        result[0:h_left, 0:w_left] = img_left
        
        # Crop the black regions from the stitched image.
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            result = result[y:y+h, x:x+w]
            
        return result
    
    def _draw_sidebar(self, frame):
        """Draws the help text sidebar."""
        y0 = 40
        for i, line in enumerate(self.help_text):
            y = y0 + i * 25
            draw_text(frame, line, pos=(10, y), scale=0.5, color=(0, 255, 0))
        return frame

    def run(self):
        """Main application loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
            output = self._process_frame(frame.copy())
            
            if self.mode != 'calibration_result':
                output = self._draw_sidebar(output)
            
            if self.stitch_status_message and time.time() - self.stitch_status_message_time < 3:
                draw_text(output, self.stitch_status_message, (50, 90), 0.8, (0, 0, 255))
            else:
                self.stitch_status_message = ""
            
            if len(self.panorama_frames) > 0 and self.mode not in ["panorama", "ar", "calibration_result"]:
                status_text = f"Frames: {len(self.panorama_frames)}. Press 'p' to stitch."
                draw_text(output, status_text, pos=(50, 50), scale=0.8, color=(255, 0, 0))

            cv2.imshow(self.window_name, output)
            key = cv2.waitKey(1) & 0xFF
            if self._handle_key_press(key, frame): break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        """Applies the currently selected mode's effect to the frame."""
        mode_handlers = {
            'gray': lambda f: cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
            'hsv': lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2HSV),
            'gaussian': self._handle_gaussian,
            'bilateral': self._handle_bilateral,
            'canny': lambda f: cv2.cvtColor(cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR),
            'hough': self._handle_hough,
            'translate': lambda f: cv2.warpAffine(f, np.float32([[1, 0, 50], [0, 1, 30]]), (f.shape[1], f.shape[0])),
            'rotate': lambda f: cv2.warpAffine(f, cv2.getRotationMatrix2D((f.shape[1]/2, f.shape[0]/2), self.rotation_angle, 1), (f.shape[1], f.shape[0])),
            'scale': lambda f: cv2.resize(f, None, fx=1.5, fy=1.5),
            'calibrate': self._handle_calibration,
            'calibration_result': self._handle_calibration_result_display,
            'undistort': self._handle_undistort,
            'panorama': self._handle_panorama_display,
            'ar': self._draw_ar_mode,
            'raw': lambda f: f
        }
        return mode_handlers.get(self.mode, lambda f: f)(frame)

    def _handle_key_press(self, key, frame):
        """Handles all user keyboard input."""
        if key == ord('q'): return True

        mode_map = {'0': 'raw', '1': 'gray', '2': 'hsv', 'g': 'gaussian', 'f': 'bilateral',
                    'e': 'canny', 'l': 'hough', 't': 'translate', 's': 'scale',
                    'k': 'calibrate', 'u': 'undistort', 'a': 'ar'}
        
        # Use a try-except block to handle non-character keys
        try:
            char_key = chr(key)
            if char_key in mode_map:
                if char_key == 'k':
                    self.images_captured, self.objpoints, self.imgpoints = 0, [], []
                    print("\nCalibration mode started.")
                self.mode = mode_map[char_key]
        except ValueError:
            pass # Key is not a standard character, ignore for mode map

        if key == ord('r'):
            self.mode = 'rotate'
            self.rotation_angle = (self.rotation_angle + 90) % 360
            print(f"Rotation set to {self.rotation_angle} degrees.")
        elif key == ord('c'): self.contrast += 0.1
        elif key == ord('v'): self.contrast = max(0.1, self.contrast - 0.1)
        elif key == ord('b'): self.brightness += 5
        elif key == ord('n'): self.brightness -= 5
        elif key == ord('h'): self._show_histogram(frame)
        elif key == ord('o'):
            self.panorama_frames.append(frame.copy())
            print(f"Image captured for panorama. Total: {len(self.panorama_frames)}")
        elif key == ord('p'): self._stitch_panorama()
        elif key == ord('z'):
            self.mode, self.panorama_base, self.panorama_frames = 'raw', None, []
            print("Panorama reset.")
        # Filter parameter controls
        elif key == ord('['): self.gaussian_kernel_size = max(1, self.gaussian_kernel_size - 2)
        elif key == ord(']'): self.gaussian_kernel_size += 2
        elif key == ord(','): self.bilateral_d = max(1, self.bilateral_d - 1) # <
        elif key == ord('.'): self.bilateral_d += 1 # >
            
        return False

    def _handle_gaussian(self, frame):
        """Applies Gaussian blur with adjustable kernel size."""
        draw_text(frame, f"Kernel: {self.gaussian_kernel_size}", (250, 40), scale=0.6)
        return cv2.GaussianBlur(frame, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
        
    def _handle_bilateral(self, frame):
        """Applies Bilateral filter with adjustable diameter."""
        draw_text(frame, f"Diameter: {self.bilateral_d}", (250, 40), scale=0.6)
        return cv2.bilateralFilter(frame, self.bilateral_d, 75, 75)

    def _handle_hough(self, frame):
        """Draws Hough lines on the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return frame

    def _show_histogram(self, frame):
        """Calculates and displays the histogram of the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
        cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
        for x in range(1, 256):
            cv2.line(hist_img, (x - 1, 300 - int(hist[x - 1])), (x, 300 - int(hist[x])), (0, 255, 0), 1)
        cv2.imshow("Histogram", hist_img)

    def _stitch_panorama(self):
        """Stitches captured frames into a panorama."""
        if len(self.panorama_frames) < 2:
            print("Not enough images. Capture at least 2 with 'o'.")
            return
        print("Starting panorama stitching...")
        
        # Start with the first image.
        self.panorama_base = self.panorama_frames[0]
        
        # Iteratively stitch the remaining images.
        for i in range(1, len(self.panorama_frames)):
            print(f"Stitching image {i+1}/{len(self.panorama_frames)}...")
            new_frame = self.panorama_frames[i]
            
            # Stitch the existing panorama (left) with the new frame (right).
            stitched_result = self._stitch_images(self.panorama_base, new_frame)
            
            if stitched_result is not None:
                self.panorama_base = stitched_result
            else:
                self.stitch_status_message = "Stitch failed. Try more overlap."
                self.stitch_status_message_time = time.time()
                print("Stitch failed. Skipping frame.")
                # We stop stitching if one fails, as subsequent ones are likely to fail too.
                break

        print("Panorama stitching complete.")
        self.mode = "panorama"

    def _handle_calibration(self, frame):
        """Handles the camera calibration process."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, self.CHESSBOARD_SIZE, None)
        if ret_corners:
            cv2.drawChessboardCorners(frame, self.CHESSBOARD_SIZE, corners, ret_corners)
            if time.time() - self.last_capture_time > 2 and self.images_captured < self.TARGET_IMAGES:
                print(f"Capturing image {self.images_captured + 1}/{self.TARGET_IMAGES}...")
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)
                self.images_captured += 1
                self.last_capture_time = time.time()
        
        draw_text(frame, f"Calibration: {self.images_captured}/{self.TARGET_IMAGES}", (50, 50), 0.8, (0,0,255))
        
        if self.images_captured >= self.TARGET_IMAGES:
            print("\nPerforming calibration...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            
            if ret:
                # Calculate re-projection error
                mean_error = 0
                for i in range(len(self.objpoints)):
                    imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error
                error_val = mean_error / len(self.objpoints)
                
                # Store results for on-screen display and console
                self.calibration_results = {'mtx': mtx, 'dist': dist, 'error': error_val}
                self.mtx, self.dist = mtx, dist # Update class instance
                
                # Print to console
                print("\nCalibration successful!")
                print(f"Total re-projection error: {error_val:.4f}")
                
                # Save the calibration result
                np.savez('calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
                print('Calibration data saved to "calibration.npz"')
                
                self.mode = "calibration_result" # Switch to display mode
            else:
                print("Calibration failed. Please try again.")
                self.mode = "raw"
        return frame

    def _handle_calibration_result_display(self, frame):
        """Displays the results of the calibration on screen."""
        # Create a black background to display text clearly
        h, w, _ = frame.shape
        display = np.zeros((h, w, 3), dtype=np.uint8)
        
        if not self.calibration_results:
            draw_text(display, "No calibration data available.", (50, 100), 0.8, (0, 0, 255))
            return display
            
        y_pos = 60
        draw_text(display, "Calibration Complete!", (50, y_pos), 1.0, (0, 255, 0))
        y_pos += 50
        
        error = self.calibration_results.get('error', 0)
        draw_text(display, f"Re-projection Error: {error:.4f}", (50, y_pos), 0.7)
        y_pos += 50
        
        draw_text(display, "Camera Matrix (mtx):", (50, y_pos), 0.7)
        y_pos += 35
        mtx = self.calibration_results.get('mtx')
        if mtx is not None:
            for i in range(mtx.shape[0]):
                row_str = " ".join([f"{val:8.2f}" for val in mtx[i]])
                draw_text(display, f"[{row_str}]", (70, y_pos), 0.6)
                y_pos += 30
        
        y_pos += 20
        draw_text(display, "Distortion Coeffs (dist):", (50, y_pos), 0.7)
        y_pos += 35
        dist = self.calibration_results.get('dist')
        if dist is not None:
             draw_text(display, str(np.round(dist, 4)), (70, y_pos), 0.6)
        
        y_pos += 70
        draw_text(display, "Press 'u' for Undistort view or any other key.", (50, y_pos), 0.7, (255, 255, 0))
        
        return display

    def _handle_undistort(self, frame):
        """Applies lens undistortion to the frame."""
        if self.mtx is None or self.dist is None:
            draw_text(frame, "Calibrate first! (Press 'k')", (50, 50), 0.8, (0,0,255))
            return frame
        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, self.mtx, self.dist, None, new_mtx)
        x, y, w, h = roi
        return undistorted[y:y+h, x:x+w]

    def _handle_panorama_display(self, frame):
        """Displays the completed panorama."""
        if self.panorama_base is not None:
            display_h = frame.shape[0]
            pano_h, pano_w = self.panorama_base.shape[:2]
            display_w = int(pano_w * (display_h / pano_h))
            display_pano = cv2.resize(self.panorama_base, (display_w, display_h))
            draw_text(display_pano, "Panorama complete. Press 'z' to reset.", (50, 50), 0.8, (0,0,255))
            return display_pano
        draw_text(frame, "Capture ('o') then stitch ('p')", (50, 50), 0.8, (0,0,255))
        return frame

    def _draw_ar_mode(self, frame):
        """Overlays a 3D model on detected ArUco markers."""
        draw_text(frame, "Mode: AUGMENTED REALITY")
        if len(self.model_vertices) == 0:
            draw_text(frame, "3D Model not loaded!", (20, 90), 0.8, (0,0,255))
            return frame
        if self.mtx is None or self.dist is None:
            draw_text(frame, "Calibrate first!", (20, 90), 0.8, (0,0,255))
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.mtx, self.dist)
            for i in range(len(ids)):
                img_pts, _ = cv2.projectPoints(self.model_vertices, rvecs[i], tvecs[i], self.mtx, self.dist)
                for face in self.model_faces:
                    points = np.int32(img_pts[face]).reshape(-1, 2)
                    cv2.polylines(frame, [points], True, (0,255,128), 1, cv2.LINE_AA)
        else:
            draw_text(frame, "No ArUco markers detected", (20, 90), 0.8, (0,0,255))
        return frame

if __name__ == '__main__':
    app = LiveCameraEffects()
    app.run()

