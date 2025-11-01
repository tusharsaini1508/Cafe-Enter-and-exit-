🚀 AI-Powered People Counter (YOLOv8) 🚀

!

This is an advanced, real-time people counting system that uses the YOLOv8 object detection model to track individuals entering and exiting an area.

The script is highly flexible, allowing you to use either a live webcam feed or a pre-recorded video file. Its key feature is an interactive setup where you dynamically draw the counting line directly on the video frame before processing begins.

✨ Features

⚡ Real-Time Tracking: Utilizes the fast and accurate YOLOv8n model for high-performance object tracking.

✏️ Interactive Line Selection: Don't hard-code coordinates! An interactive window pops up on launch, allowing you to draw your "IN/OUT" line with your mouse.

📁 Dual Video Source:

Live Webcam: Monitor a live feed from your webcam (ideal for real-world deployment).

Video File: Analyze and process any existing video file.

📊 Dynamic UI Panel: A sleek, dark overlay panel displays real-time statistics:

IN Count: Total people entering.

OUT Count: Total people exiting.

FPS: Current processing speed.

Recent Events: A live-updating log of the most recent crossings.

🧾 CSV Data Logging: Automatically saves every "IN" and "OUT" event to entry_data.csv with a precise timestamp, track ID, and direction for later analysis.

⚙️ Optimized Performance: Resizes frames for faster inference (INFERENCE_WIDTH) while scaling coordinates back for accurate tracking on the original video.

🛠️ How to Use

1. Prerequisites

Make sure you have Python installed, along with the required libraries.

pip install opencv-python ultralytics


You will also need the YOLOv8 model weights. The script will automatically download yolov8n.pt the first time you run it if it's not present.

2. Run the Script

From your terminal, simply run the Python file:

python people_counter.py


3. Select Video Source

You will be prompted in the terminal:

Select video source:
  1: Live Webcam
  2: Video File
Enter choice (1 or 2):


Enter 1 for your webcam.

Enter 2 for a video file (you will then be prompted to enter the file path).

4. ❗ Draw Your Line (Interactive Setup)

A window titled "Select Counting Line" will open, showing the first frame of your video.

Click once to set the first point of your line.

Click a second time to set the second point.

A red line will appear.

Press c to confirm the line and start processing.

Press r to reset the points and draw again.

Press q to quit the setup.

5. View the Results

A new window, "People IN/OUT Counter," will open. You will see the live video with:

Green bounding boxes around detected people.

Your purple counting line.

The real-time data panel on the right.

⌨️ Controls (During Processing)

q: Press 'q' at any time to quit the application.

r: Press 'r' to reset the "IN" and "OUT" counts to zero.

📄 License

This project is licensed under the Tushar Saini License.

The Tushar Saini License (TSL 1.0)

© 2025, Tushar Saini. All Rights Reserved.

This software and associated documentation files (the "Software") are the exclusive property of Tushar Saini.

You are hereby granted a non-exclusive, non-transferable license to:

Use: You may use the Software for personal, educational, and internal business purposes.

Modify: You may modify the Software for your own personal, educational, or internal business use. All modifications remain subject to this license.

You are strictly prohibited from:

Redistribution: You may not redistribute, sublicense, sell, rent, lease, or otherwise make the Software available to any third party, in whole or in part, with or without modification.

Public Deployment: You may not deploy this Software in a public-facing commercial application or service without the express written permission of Tushar Saini.

Removal of Notices: You may not remove or alter any copyright, trademark, or other proprietary notices contained in the Software.

Disclaimer of Warranty:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contact:

For any inquiries, permissions, or commercial licensing requests, please contact Tushar Saini.
