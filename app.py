import streamlit as st
import pandas as pd
import sqlite3
import qrcode
import cv2
import numpy as np
import json
import uuid

from pathlib import Path
from datetime import datetime
from PIL import Image
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="E-Waste Detection and Recycling Tracker",
    page_icon="♻️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "ewaste_tracker.db"
QR_DIR = BASE_DIR / "qrs"
CROP_DIR = BASE_DIR / "crops"

QR_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "best.pt"

STATUSES = [
    "Detected",
    "Sorted",
    "Recycled",
    "Disposed"
]

EWASTE_WORDS = [
    "battery", "pcb", "circuit", "mobile",
    "phone", "laptop", "computer", "keyboard",
    "mouse", "printer", "television", "tv",
    "monitor", "microwave", "washing", "player",
    "charger", "cable", "electronic", "tablet",
    "camera", "remote", "fan"
]


# ==========================================
# 2. DATABASE
# ==========================================

def create_database():

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                qr_path TEXT NOT NULL,
                crop_path TEXT
            )
        """)

        conn.commit()


def get_all_items():

    with sqlite3.connect(DB_PATH) as conn:

        df = pd.read_sql_query("""
            SELECT
                item_id,
                item_type,
                status,
                confidence,
                created_at,
                qr_path,
                crop_path
            FROM items
            ORDER BY created_at DESC
        """, conn)

    return df


def get_item(item_id):

    with sqlite3.connect(DB_PATH) as conn:

        return conn.execute("""
            SELECT *
            FROM items
            WHERE item_id = ?
        """, (item_id,)).fetchone()


# ==========================================
# 3. GENERATE UNIQUE QR CODE FOR EACH ITEM
# ==========================================

def save_item(item_type, confidence, crop_path=None):

    # Generate a unique ID for every item
    item_id = "EW-" + uuid.uuid4().hex[:10].upper()

    if "battery" in item_type.lower():
        status = "Disposed"
    else:
        status = "Detected"

    # Unique QR image filename
    qr_path = QR_DIR / f"{item_id}.png"

    # QR contains this item's identity
    qr_data = {
        "item_id": item_id,
        "item_type": item_type
    }

    qr_image = qrcode.make(json.dumps(qr_data))
    qr_image.save(qr_path)

    # Save individual item details
    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            INSERT INTO items (
                item_id,
                item_type,
                status,
                confidence,
                created_at,
                qr_path,
                crop_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id,
            item_type,
            status,
            float(confidence),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(qr_path.relative_to(BASE_DIR)),
            str(crop_path.relative_to(BASE_DIR))
            if crop_path else None
        ))

        conn.commit()

    return item_id, status, qr_path


# ==========================================
# 4. UPDATE ITEM STATUS
# ==========================================

def update_status(item_id, status):

    with sqlite3.connect(DB_PATH) as conn:

        cursor = conn.execute("""
            UPDATE items
            SET status = ?
            WHERE item_id = ?
        """, (status, item_id))

        conn.commit()

        return cursor.rowcount > 0


# ==========================================
# 5. LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))

    return YOLO("yolov8n.pt")


try:
    model = load_model()

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# ==========================================
# 6. DETECT E-WASTE
# ==========================================

def detect_objects(image, confidence_range):

    results = model.predict(
        source=image,
        conf=confidence_range,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is None:
        return result, detections

    image_array = np.array(image.convert("RGB"))

    height, width = image_array.shape[:2]

    for box in result.boxes:

        class_id = int(box.cls[0].item())

        confidence = float(box.conf[0].item())

        label = str(model.names[class_id])

        # Check whether the detected class is e-waste
        if not any(
            word in label.lower()
            for word in EWASTE_WORDS
        ):
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))

        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        crop = image_array[y1:y2, x1:x2]

        crop_path = None

        if crop.size > 0:

            crop_path = CROP_DIR / (
                uuid.uuid4().hex[:10] + ".jpg"
            )

            Image.fromarray(crop).save(crop_path)

        detections.append({
            "label": label,
            "confidence": confidence,
            "crop_path": crop_path
        })

    return result, detections


# ==========================================
# 7. QR CODE SCANNER
# ==========================================

def scan_qr(image):

    image_array = np.array(image.convert("RGB"))

    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    detector = cv2.QRCodeDetector()

    data, points, _ = detector.detectAndDecode(image_bgr)

    return data if data else None


# ==========================================
# 8. INITIALIZE
# ==========================================

create_database()

st.sidebar.title("♻️ E-Waste Tracker")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "E-Waste Detection",
        "QR Code Scanner",
        "Update Status",
        "Dashboard",
        "Records",
        "Reports"
    ]
)


# ==========================================
# 9. HOME PAGE
# ==========================================

if page == "Home":

    st.title("♻️ E-Waste Detection and Recycling Tracker")

    st.write("""
    An Intelligent Embedded System for Automated
    E-Waste Classification, Segregation, and
    Lifecycle Tracking Using Machine Vision
    and QR-Based Traceability.
    """)

    st.subheader("Project Features")

    st.markdown("""
    - YOLO-based E-Waste Detection
    - Confidence Range Selection
    - Unique QR Code Generation
    - Individual QR Code Download
    - QR Code Scanning
    - Recycling Status Tracking
    - SQLite Database
    - Dashboard and Reports
    """)


# ==========================================
# 10. E-WASTE DETECTION PAGE
# ==========================================

elif page == "E-Waste Detection":

    st.title("E-Waste Detection")

    input_type = st.radio(
        "Select Image Source",
        ["Upload Image", "Camera"],
        horizontal=True
    )

    uploaded_image = None

    if input_type == "Upload Image":

        file = st.file_uploader(
            "Upload E-Waste Image",
            type=["jpg", "jpeg", "png"]
        )

        if file:
            uploaded_image = Image.open(file).convert("RGB")

    else:

        camera_file = st.camera_input(
            "Capture E-Waste Image"
        )

        if camera_file:
            uploaded_image = Image.open(camera_file).convert("RGB")

    # Confidence slider
    st.subheader("Confidence Range")

    confidence_range = st.slider(
        "Adjust Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05
    )

    st.metric(
        "Selected Confidence",
        f"{confidence_range * 100:.0f}%"
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Detect E-Waste", type="primary"):

            with st.spinner("Detecting E-Waste..."):

                result, detections = detect_objects(
                    uploaded_image,
                    confidence_range
                )

                annotated = result.plot()

                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                st.session_state["detections"] = detections
                st.session_state["annotated"] = annotated

                # Clear old registration state for a new detection
                st.session_state["registered_items"] = {}

        if "annotated" in st.session_state:

            st.subheader("Detection Results")

            st.image(
                st.session_state["annotated"],
                use_container_width=True
            )

            detections = st.session_state["detections"]

            if not detections:

                st.warning(
                    "No E-Waste detected. Check your trained model."
                )

            else:

                for index, detection in enumerate(detections):

                    st.divider()

                    st.subheader(
                        f"Item {index + 1}: {detection['label']}"
                    )

                    st.write(
                        "Confidence:",
                        f"{detection['confidence'] * 100:.2f}%"
                    )

                    st.progress(detection["confidence"])

                    registered = st.session_state[
                        "registered_items"
                    ]

                    if index not in registered:

                        if st.button(
                            f"Generate QR for Item {index + 1}",
                            key=f"register_{index}"
                        ):

                            item_id, status, qr_path = save_item(
                                detection["label"],
                                detection["confidence"],
                                detection["crop_path"]
                            )

                            st.session_state[
                                "registered_items"
                            ][index] = {
                                "item_id": item_id,
                                "status": status,
                                "qr_path": str(qr_path)
                            }

                            st.rerun()

                    else:

                        saved = registered[index]

                        st.success("Unique QR Code Generated!")

                        st.write("Item ID:", saved["item_id"])
                        st.write("Status:", saved["status"])

                        qr_bytes = Path(
                            saved["qr_path"]
                        ).read_bytes()

                        st.image(
                            qr_bytes,
                            caption="Individual QR Code",
                            width=180
                        )

                        st.download_button(
                            "Download This Item's QR Code",
                            data=qr_bytes,
                            file_name=f"{saved['item_id']}.png",
                            mime="image/png",
                            key=f"download_{saved['item_id']}"
                        )


# ==========================================
# 11. QR CODE SCANNER PAGE
# ==========================================

elif page == "QR Code Scanner":

    st.title("QR Code Scanner")

    qr_file = st.file_uploader(
        "Upload QR Code Image",
        type=["png", "jpg", "jpeg"]
    )

    if qr_file:

        qr_image = Image.open(qr_file).convert("RGB")

        st.image(qr_image, width=200)

        if st.button("Scan QR Code"):

            data = scan_qr(qr_image)

            if data:

                try:

                    qr_data = json.loads(data)

                    item_id = qr_data["item_id"]

                    item = get_item(item_id)

                    if item:

                        st.success("QR Code Scanned Successfully!")

                        st.write("Item ID:", item[0])
                        st.write("Item Type:", item[1])
                        st.write("Current Status:", item[2])

                        st.write(
                            "Confidence:",
                            f"{item[3] * 100:.2f}%"
                        )

                        st.write("Created At:", item[4])

                    else:

                        st.error("Item not found in database.")

                except Exception:

                    st.error("Invalid QR Code.")

            else:

                st.error("QR Code not detected.")


# ==========================================
# 12. UPDATE STATUS PAGE
# ==========================================

elif page == "Update Status":

    st.title("Update Recycling Status")

    df = get_all_items()

    if not df.empty:

        item_ids = df["item_id"].tolist()

        item_id = st.selectbox(
            "Select Item ID",
            item_ids
        )

        item = get_item(item_id)

        if item:

            st.write("Item Type:", item[1])
            st.write("Current Status:", item[2])

            new_status = st.selectbox(
                "New Status",
                STATUSES,
                index=STATUSES.index(item[2])
            )

            if st.button("Update Status"):

                if update_status(item_id, new_status):

                    st.success("Status Updated Successfully!")

                    st.rerun()

    else:

        st.info("No items available.")


# ==========================================
# 13. DASHBOARD
# ==========================================

elif page == "Dashboard":

    st.title("E-Waste Dashboard")

    df = get_all_items()

    if not df.empty:

        df["confidence"] = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )

        total = len(df)

        recycled = len(df[df["status"] == "Recycled"])

        disposed = len(df[df["status"] == "Disposed"])

        sorted_items = len(df[df["status"] == "Sorted"])

        avg_confidence = df["confidence"].mean()

        if pd.isna(avg_confidence):
            avg_confidence = 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Items", total)
        col2.metric("Recycled", recycled)
        col3.metric("Disposed", disposed)
        col4.metric("Sorted", sorted_items)

        st.subheader("Average Confidence")

        st.metric(
            "Average Confidence",
            f"{avg_confidence * 100:.2f}%"
        )

        st.subheader("E-Waste Type Distribution")

        st.bar_chart(
            df["item_type"].value_counts()
        )

        st.subheader("Status Distribution")

        st.bar_chart(
            df["status"].value_counts()
        )

        st.subheader("Confidence by Item")

        chart_df = df[
            ["item_id", "confidence"]
        ].set_index("item_id") * 100

        st.bar_chart(chart_df)

    else:

        st.info("No records available.")


# ==========================================
# 14. RECORDS PAGE - SEPARATE QR DOWNLOADS
# ==========================================

elif page == "Records":

    st.title("E-Waste Records")

    df = get_all_items()

    if not df.empty:

        st.write("Total Records:", len(df))

        st.dataframe(
            df[[
                "item_id",
                "item_type",
                "status",
                "confidence",
                "created_at"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Individual QR Codes")

        for _, row in df.iterrows():

            with st.container(border=True):

                col1, col2 = st.columns([2, 1])

                with col1:

                    st.write("**Item ID:**", row["item_id"])

                    st.write(
                        "**E-Waste Type:**",
                        row["item_type"]
                    )

                    st.write("**Status:**", row["status"])

                    st.write(
                        "**Confidence:**",
                        f"{row['confidence'] * 100:.2f}%"
                    )

                with col2:

                    qr_path = Path(str(row["qr_path"]))

                    if not qr_path.is_absolute():
                        qr_path = BASE_DIR / qr_path

                    if qr_path.exists():

                        qr_bytes = qr_path.read_bytes()

                        st.image(qr_bytes, width=120)

                        st.download_button(
                            "Download QR Code",
                            data=qr_bytes,
                            file_name=f"{row['item_id']}.png",
                            mime="image/png",
                            key=f"record_download_{row['item_id']}"
                        )

                    else:

                        st.error("QR image not found.")


    else:

        st.info("No records available.")


# ==========================================
# 15. REPORTS PAGE
# ==========================================

elif page == "Reports":

    st.title("E-Waste Reports")

    df = get_all_items()

    if not df.empty:

        report_df = df.drop(
            columns=["qr_path", "crop_path"]
        )

        report_df = report_df.rename(columns={
            "item_id": "Item ID",
            "item_type": "E-Waste Type",
            "status": "Status",
            "confidence": "Confidence",
            "created_at": "Created At"
        })

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = report_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download CSV Report",
            data=csv_data,
            file_name="ewaste_report.csv",
            mime="text/csv"
        )

    else:

        st.info("No reports available.")


# ==========================================
# 16. FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "E-Waste Detection and Recycling Tracker"
)

st.sidebar.caption(
    "Powered by Python, YOLO and Streamlit"
)

st.divider()

st.caption(
    "Machine Vision and QR-Based Traceability"
)