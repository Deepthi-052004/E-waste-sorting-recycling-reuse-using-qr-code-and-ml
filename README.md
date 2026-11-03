# E-waste-sorting-recycling-reuse-using-qr-code-and-ml
♻️ E-Waste Detection and Recycling Tracker
An Intelligent Embedded System for Automated E-Waste Classification, Segregation, and Lifecycle Tracking Using Machine Vision and QR-Based Traceability
About the Project
The E-Waste Detection and Recycling Tracker is an AI-powered application
developed to identify and track electronic waste using Machine Learning
and Computer Vision.

The system uses the YOLO object detection model to detect electronic
waste from images. It generates a unique QR code for each detected item
and stores the information in an SQLite database.
Objectives
- Automate electronic waste detection.
- Classify different types of e-waste.
- Generate unique QR codes for detected items.
- Track the lifecycle of electronic waste.
- Maintain electronic waste records.
- Support better recycling and disposal management.
Key Features
- YOLO-based object detection
- Image upload and camera capture
- QR code generation and scanning
- Individual QR code downloads from the records table
- Recycling status tracking
- SQLite database management
- Dashboard and detection statistics
- CSV report downloads
Technologies Used
  Technology   Application
  Python       Backend programming
YOLO         Object detection
OpenCV       Image processing
Streamlit    Web application
SQLite       Database
Pandas       Data management
QR Code      Item tracking
Pillow       Image handling
Project Structure
E-Waste-Tracker/
├── app.py
├── best.pt
├── ewaste_v2.db
├── README.md
├── qrs/
│   └── Generated QR codes
└── crops/
    └── Detected object images
Installation
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
2. Navigate to the project folder
cd E-Waste-Tracker
3. Install dependencies
pip install streamlit ultralytics opencv-python pillow pandas qrcode
4. Add the trained YOLO model
Place your custom-trained best.pt file inside the project folder.
5. Run the application
streamlit run app.py
Open the local URL shown in your terminal, usually
http://localhost:8501.
Project Workflow
1. Upload an e-waste image or capture an image using the camera.
2. YOLO processes the image and detects objects.
3. The application displays detected objects and confidence scores.
4. A unique item ID and QR code are generated.
5. Item details are stored in the SQLite database.
6. Scan the QR code to retrieve item information.
7. Update the recycling or disposal status.
8. View records and statistics on the dashboard.
9. Download individual QR codes and CSV reports.
Application Modules
1. Home
2. E-Waste Detection
3. QR Code Scanner
4. Update Recycling Status
5. Dashboard
6. E-Waste Records
7. Reports
Future Enhancements
- Real-time video-based detection
- Improved detection accuracy using custom datasets
- IoT-based smart waste collection
- Cloud database integration
- Automated hazardous waste alerts
- Integration with recycling centres
Project Domain
Artificial Intelligence | Machine Learning | Computer Vision |
Sustainable Environment | Waste Management
Smart Technology for a Sustainable Environment!
