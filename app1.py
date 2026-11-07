import streamlit as st
import cv2
import time
from pathlib import Path
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Smart Traffic Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ATTRACTIVE LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>
    /* ===== GLOBAL ===== */
    .stApp {
        background: #f4f7fb;
        color: #172033;
    }

    .main .block-container {
        max-width: 1500px;
        padding: 0.25rem 1.25rem 0.7rem 1.25rem;
    }

    /* Remove the dark strip at the top and keep the page light */
    header[data-testid="stHeader"],
    .stAppHeader {
        background: #f4f7fb !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #172033 !important;
    }

    p, li, label, span {
        color: #263247;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f6fb 100%);
        border-right: 2px solid #cbd8e6;
    }

    section[data-testid="stSidebar"] > div {
        padding: 0 !important;
    }

    /* Keep sidebar content tidy while letting the brand fill the top area */
    section[data-testid="stSidebar"] .block-container {
        padding: 0.65rem 0.7rem 0.45rem 0.7rem !important;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0.12rem;
    }

    section[data-testid="stSidebar"] .stRadio label {
        border: 1px solid #d9e3ee;
        border-radius: 8px;
        padding: 0.22rem 0.45rem;
        background: #ffffff;
    }

    section[data-testid="stSidebar"] hr {
        margin: 0.45rem 0;
        border-color: #d2dce8;
    }

    .side-brand {
        text-align: center;
        background: linear-gradient(135deg, #e4effa 0%, #ffffff 55%, #eaf4fc 100%);
        border-bottom: 2px solid #cbd8e6;
        margin: -0.65rem -0.7rem 0.65rem -0.7rem;
        padding: 1.35rem 0.45rem 1.1rem 0.45rem;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }

    .side-icon {
        font-size: 2.5rem;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }

    .side-title {
        color: #17365d !important;
        font-size: 1.55rem;
        line-height: 1.15;
        font-weight: 900;
        letter-spacing: 0.025em;
        margin-top: 0.15rem;
        white-space: nowrap;
    }

    .side-subtitle {
        color: #617087 !important;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        width: 100%;
        min-height: 40px;
        border-radius: 9px;
        border: 1.5px solid #b9c9da;
        background: #ffffff;
        color: #17365d;
        font-weight: 800;
        box-shadow: 0 2px 5px rgba(33, 58, 88, 0.06);
    }

    .stButton > button:hover {
        border-color: #54799f;
        background: #edf4fb;
        color: #17365d;
    }

    /* ===== DASHBOARD HEADER ===== */
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #edf5fc 100%);
        border: 1.5px solid #c5d7e8;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 3px 10px rgba(40, 67, 95, 0.07);
    }

    .hero-title {
        text-align: center;
        color: #17365d !important;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 0.02em;
    }

    .hero-subtitle {
        text-align: center;
        color: #64748b !important;
        font-size: 0.86rem;
        margin-top: 0.1rem;
    }

    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: #ffffff;
        border: 1.5px solid #cbd8e6;
        border-radius: 11px;
        padding: 0.55rem 0.4rem;
        text-align: center;
        min-height: 82px;
        box-shadow: 0 2px 7px rgba(39, 65, 91, 0.06);
    }

    .metric-card.green {
        border-top: 4px solid #2e9b62;
    }

    .metric-card.blue {
        border-top: 4px solid #4c78a8;
    }

    .metric-card.orange {
        border-top: 4px solid #d99028;
    }

    .metric-card.purple {
        border-top: 4px solid #8067b5;
    }

    .metric-label {
        color: #68768a !important;
        font-size: 0.67rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    .metric-value {
        color: #172033 !important;
        font-size: 1.28rem;
        font-weight: 900;
        margin-top: 0.12rem;
    }

    /* ===== SIGNAL ===== */
    .signal-panel {
        background: linear-gradient(135deg, #f2fbf6 0%, #e8f7ef 100%);
        border: 1.5px solid #9bd1b2;
        border-left: 6px solid #2e9b62;
        border-radius: 11px;
        padding: 0.6rem 0.8rem;
        margin: 0.65rem 0 0.75rem 0;
        text-align: center;
        box-shadow: 0 2px 7px rgba(46, 155, 98, 0.07);
    }

    .signal-heading {
        color: #3f6d52 !important;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }

    .signal-name {
        color: #177245 !important;
        font-size: 1.5rem;
        font-weight: 900;
    }

    .signal-details {
        color: #52695a !important;
        font-size: 0.78rem;
    }

    /* ===== STATUS ===== */
    .status-running {
        background: #effaf4;
        border: 1.5px solid #9bd1b2;
        color: #177245 !important;
        border-radius: 8px;
        padding: 0.42rem;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }

    .status-stopped {
        background: #fff9ec;
        border: 1.5px solid #ead49e;
        color: #765b17 !important;
        border-radius: 8px;
        padding: 0.42rem;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }

    /* ===== SECTION HEADERS ===== */
    .section-heading {
        color: #17365d !important;
        font-size: 1.08rem;
        font-weight: 900;
        border-left: 5px solid #4c78a8;
        border-bottom: 1px solid #d6e0eb;
        padding: 0.25rem 0 0.28rem 0.55rem;
        margin: 0.65rem 0 0.5rem 0;
    }

    /* ===== CAMERA ===== */
    .camera-card {
        background: #ffffff;
        border: 1.5px solid #c8d6e5;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 3px 9px rgba(38, 61, 85, 0.07);
        margin-bottom: 0.35rem;
    }

    .camera-title {
        background: linear-gradient(90deg, #eaf2fa, #f7faff);
        color: #17365d !important;
        border-bottom: 1.5px solid #c8d6e5;
        text-align: center;
        padding: 0.34rem;
        font-size: 0.82rem;
        font-weight: 900;
        letter-spacing: 0.03em;
    }

    /* ===== INFO / ABOUT CARDS ===== */
    .info-panel {
        background: #ffffff;
        border: 1.5px solid #cbd8e6;
        border-radius: 11px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 8px rgba(39, 65, 91, 0.05);
        margin-bottom: 0.7rem;
    }

    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        color: #7b8798 !important;
        font-size: 0.7rem;
        padding: 0.7rem 0 0.1rem 0;
        border-top: 1px solid #d7e0ea;
        margin-top: 0.8rem;
    }

    /* ===== DATAFRAME ===== */
    div[data-testid="stDataFrame"] {
        border: 1.5px solid #cbd8e6;
        border-radius: 9px;
    }

    /* Compact spacing */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.42rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"
MODEL_PATH = BASE_DIR / "yolo11n.pt"
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH = RESULTS_DIR / "traffic_data.csv"

DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]

VIDEOS = {
    "NORTH": VIDEO_DIR / "north.mp4",
    "SOUTH": VIDEO_DIR / "south.mp4",
    "EAST": VIDEO_DIR / "east.mp4",
    "WEST": VIDEO_DIR / "west.mp4",
}

VEHICLE_CLASSES = [2, 3, 5, 7]

HIGH_TIME = 60
MEDIUM_TIME = 45
LOW_TIME = 30
HIGH_THRESHOLD = 15
MEDIUM_THRESHOLD = 7


# ============================================================
# STATE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "caps" not in st.session_state:
    st.session_state.caps = {}

if "counts" not in st.session_state:
    st.session_state.counts = {d: 0 for d in DIRECTIONS}

if "current_signal" not in st.session_state:
    st.session_state.current_signal = "NORTH"

if "signal_index" not in st.session_state:
    st.session_state.signal_index = 0

if "signal_duration" not in st.session_state:
    st.session_state.signal_duration = HIGH_TIME

if "signal_start" not in st.session_state:
    st.session_state.signal_start = time.time()

if "signal_initialized" not in st.session_state:
    st.session_state.signal_initialized = False

if "last_saved_second" not in st.session_state:
    st.session_state.last_saved_second = -1


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def get_traffic_level(count):
    if count >= HIGH_THRESHOLD:
        return "HIGH", HIGH_TIME
    if count >= MEDIUM_THRESHOLD:
        return "MEDIUM", MEDIUM_TIME
    return "LOW", LOW_TIME


def release_cameras():
    for cap in st.session_state.caps.values():
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.caps = {}


def initialize_cameras():
    if len(st.session_state.caps) == 4:
        return True

    release_cameras()

    for direction, path in VIDEOS.items():
        if not path.exists():
            continue

        cap = cv2.VideoCapture(str(path))

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if cap.isOpened():
            st.session_state.caps[direction] = cap
        else:
            cap.release()

    return len(st.session_state.caps) == 4


def set_signal(direction):
    index = DIRECTIONS.index(direction)
    count = st.session_state.counts.get(direction, 0)
    _, duration = get_traffic_level(count)

    st.session_state.signal_index = index
    st.session_state.current_signal = direction
    st.session_state.signal_duration = duration
    st.session_state.signal_start = time.time()


def move_to_next_signal():
    next_index = (st.session_state.signal_index + 1) % len(DIRECTIONS)
    set_signal(DIRECTIONS[next_index])


def process_frames(model):
    frames = {}
    counts = {}

    for direction in DIRECTIONS:
        cap = st.session_state.caps.get(direction)

        if cap is None:
            continue

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if not ret:
            continue

        try:
            result = model(frame, conf=0.25, verbose=False)[0]
        except Exception:
            continue

        vehicle_count = 0

        if result.boxes is not None:
            classes = result.boxes.cls.int().cpu().tolist()
            vehicle_count = sum(
                1 for class_id in classes if class_id in VEHICLE_CLASSES
            )

        counts[direction] = vehicle_count

        annotated = result.plot()

        # Readable labels on the video itself.
        cv2.putText(
            annotated,
            direction,
            (15, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            annotated,
            direction,
            (15, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            f"Vehicles: {vehicle_count}",
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            annotated,
            f"Vehicles: {vehicle_count}",
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

        frames[direction] = annotated

    return frames, counts


def save_result():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    current = st.session_state.current_signal
    count = st.session_state.counts.get(current, 0)
    level, duration = get_traffic_level(count)

    row = {
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "North Vehicles": st.session_state.counts["NORTH"],
        "South Vehicles": st.session_state.counts["SOUTH"],
        "East Vehicles": st.session_state.counts["EAST"],
        "West Vehicles": st.session_state.counts["WEST"],
        "Green Direction": current,
        "Traffic Level": level,
        "Green Duration": duration,
    }

    new_df = pd.DataFrame([row])

    if CSV_PATH.exists():
        try:
            old_df = pd.read_csv(CSV_PATH)
            final_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            final_df = new_df
    else:
        final_df = new_df

    final_df.to_csv(CSV_PATH, index=False)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="side-icon">🚦</div>
            <div class="side-title">SMART TRAFFIC</div>
            <div class="side-subtitle">AI Traffic Management</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Live Dashboard",
            "📊 Analytics",
            "📋 Traffic Data",
            "🚦 Signal Information",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**System Control**")

    start_clicked = st.button("▶ START SYSTEM", use_container_width=True)
    stop_clicked = st.button("■ STOP SYSTEM", use_container_width=True)

    if start_clicked:
        if not MODEL_PATH.exists():
            st.error("yolo11n.pt was not found.")
        elif initialize_cameras():
            st.session_state.running = True
            st.session_state.signal_index = 0
            st.session_state.current_signal = "NORTH"
            st.session_state.signal_initialized = False
            st.session_state.signal_start = time.time()
            st.rerun()
        else:
            st.error("Please check north.mp4, south.mp4, east.mp4 and west.mp4.")

    if stop_clicked:
        st.session_state.running = False
        release_cameras()
        st.session_state.signal_initialized = False
        st.rerun()

    st.divider()

    st.markdown("**Signal Timing**")
    st.caption("🟢 High Traffic  •  60 seconds")
    st.caption("🟡 Medium Traffic  •  45 seconds")
    st.caption("🔵 Low Traffic  •  30 seconds")

    st.divider()

    if st.session_state.running:
        st.success("SYSTEM RUNNING")
    else:
        st.info("SYSTEM STOPPED")


# ============================================================
# LIVE DASHBOARD
# ============================================================

if page == "🏠 Live Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🚦 SMART TRAFFIC MANAGEMENT</div>
            <div class="hero-subtitle">
                AI-Based Adaptive Traffic Signal Management System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.running:
        st.markdown(
            '<div class="status-stopped">🟡 System is stopped — use START SYSTEM in the sidebar.</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        cards = [
            ("ACTIVE CAMERAS", "4", "blue"),
            ("AI MODEL", "YOLO", "purple"),
            ("HIGH TRAFFIC", "60s", "orange"),
            ("MEDIUM TRAFFIC", "45s", "green"),
        ]

        for col, (label, value, cls) in zip([c1, c2, c3, c4], cards):
            with col:
                st.markdown(
                    f'<div class="metric-card {cls}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div class="info-panel">
                <h3>🚦 Four-Camera AI Monitoring</h3>
                <p>
                The system detects vehicles from North, South, East and West
                road cameras and assigns green-signal duration according to
                traffic level.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        try:
            model = load_model()
        except Exception as e:
            st.error(f"Unable to load YOLO model: {e}")
            st.stop()

        if not initialize_cameras():
            st.error("One or more camera videos could not be opened.")
            st.stop()

        # Streamlit fragment keeps the live dashboard refreshing without
        # rebuilding the entire page each frame. This is much closer to
        # the continuous behaviour seen when running the project locally.
        try:
            fragment = st.fragment
        except AttributeError:
            fragment = None

        def render_live():
            frames, counts = process_frames(model)

            for direction, count in counts.items():
                st.session_state.counts[direction] = count

            if not st.session_state.signal_initialized:
                _, duration = get_traffic_level(
                    st.session_state.counts["NORTH"]
                )
                st.session_state.signal_duration = duration
                st.session_state.signal_start = time.time()
                st.session_state.signal_initialized = True

            elapsed = int(time.time() - st.session_state.signal_start)
            remaining = st.session_state.signal_duration - elapsed

            if remaining <= 0:
                move_to_next_signal()
                remaining = st.session_state.signal_duration

            current = st.session_state.current_signal
            current_count = st.session_state.counts.get(current, 0)
            current_level, current_duration = get_traffic_level(current_count)

            # Save no more than one row per second.
            second_now = int(time.time())
            if second_now != st.session_state.last_saved_second:
                save_result()
                st.session_state.last_saved_second = second_now

            # Metrics
            c1, c2, c3, c4 = st.columns(4)

            total_vehicles = sum(st.session_state.counts.values())

            metric_data = [
                ("ACTIVE CAMERAS", "4", "blue"),
                ("CURRENT GREEN", current, "green"),
                ("TOTAL VEHICLES", str(total_vehicles), "orange"),
                ("TIME LEFT", f"{remaining}s", "purple"),
            ]

            for col, (label, value, cls) in zip(
                [c1, c2, c3, c4], metric_data
            ):
                with col:
                    st.markdown(
                        f'<div class="metric-card {cls}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown(
                f"""
                <div class="signal-panel">
                    <div class="signal-heading">CURRENT GREEN SIGNAL</div>
                    <div class="signal-name">🟢 {current}</div>
                    <div class="signal-details">
                        {remaining} seconds remaining
                        &nbsp; • &nbsp; {current_level} traffic
                        &nbsp; • &nbsp; {current_duration}s allocated
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="status-running">🟢 AI vehicle detection is active • Four traffic cameras are being monitored</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-heading">Traffic Status by Direction</div>',
                unsafe_allow_html=True,
            )

            cols = st.columns(4)

            for col, direction in zip(cols, DIRECTIONS):
                count = st.session_state.counts.get(direction, 0)
                level, duration = get_traffic_level(count)

                with col:
                    st.metric(direction, count)
                    st.caption(f"{level} Traffic • {duration}s")

            st.markdown(
                '<div class="section-heading">🎥 Four-Camera Traffic Monitoring</div>',
                unsafe_allow_html=True,
            )

            rows = [("NORTH", "SOUTH"), ("EAST", "WEST")]

            for left, right in rows:
                col1, col2 = st.columns(2)

                for col, direction in [(col1, left), (col2, right)]:
                    with col:
                        st.markdown(
                            f'<div class="camera-card"><div class="camera-title">{direction} CAMERA</div></div>',
                            unsafe_allow_html=True,
                        )

                        if direction in frames:
                            image = cv2.cvtColor(
                                frames[direction], cv2.COLOR_BGR2RGB
                            )
                            st.image(
                                image,
                                use_container_width=True,
                            )
                        else:
                            st.warning(f"{direction} video frame unavailable.")

        if fragment is not None:
            @st.fragment(run_every=0.20)
            def live_fragment():
                render_live()

            live_fragment()
        else:
            render_live()
            time.sleep(0.20)
            st.rerun()


# ============================================================
# ANALYTICS — ALL SIX GRAPHS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Traffic Analytics")

    if not CSV_PATH.exists():
        st.info("No traffic data is available yet. Start the system first.")
    else:
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception as e:
            st.error(f"Could not read traffic data: {e}")
            st.stop()

        st.success(f"Traffic data loaded successfully • {len(df)} records")

        # ----------------------------------------------------
        # 1. Average Vehicle Count
        # ----------------------------------------------------

        vehicle_columns = [
            "North Vehicles",
            "South Vehicles",
            "East Vehicles",
            "West Vehicles",
        ]

        if all(c in df.columns for c in vehicle_columns):
            st.markdown(
                '<div class="section-heading">1. Average Vehicle Count</div>',
                unsafe_allow_html=True,
            )

            avg_vehicle = pd.Series(
                {
                    "North": df["North Vehicles"].mean(),
                    "South": df["South Vehicles"].mean(),
                    "East": df["East Vehicles"].mean(),
                    "West": df["West Vehicles"].mean(),
                }
            )

            st.bar_chart(avg_vehicle, height=250)

        # ----------------------------------------------------
        # 2. Average Queue Length
        # ----------------------------------------------------

        queue_columns = [
            "North Queue",
            "South Queue",
            "East Queue",
            "West Queue",
        ]

        if all(c in df.columns for c in queue_columns):
            st.markdown(
                '<div class="section-heading">2. Average Queue Length</div>',
                unsafe_allow_html=True,
            )

            avg_queue = pd.Series(
                {
                    "North": df["North Queue"].mean(),
                    "South": df["South Queue"].mean(),
                    "East": df["East Queue"].mean(),
                    "West": df["West Queue"].mean(),
                }
            )

            st.bar_chart(avg_queue, height=250)

        # ----------------------------------------------------
        # 3. Average Waiting Time
        # ----------------------------------------------------

        waiting_columns = [
            "North Waiting",
            "South Waiting",
            "East Waiting",
            "West Waiting",
        ]

        if all(c in df.columns for c in waiting_columns):
            st.markdown(
                '<div class="section-heading">3. Average Waiting Time</div>',
                unsafe_allow_html=True,
            )

            avg_waiting = pd.Series(
                {
                    "North": df["North Waiting"].mean(),
                    "South": df["South Waiting"].mean(),
                    "East": df["East Waiting"].mean(),
                    "West": df["West Waiting"].mean(),
                }
            )

            st.bar_chart(avg_waiting, height=250)

        # ----------------------------------------------------
        # 4. Average Traffic Score
        # ----------------------------------------------------

        score_columns = [
            "North Score",
            "South Score",
            "East Score",
            "West Score",
        ]

        if all(c in df.columns for c in score_columns):
            st.markdown(
                '<div class="section-heading">4. Average Traffic Score</div>',
                unsafe_allow_html=True,
            )

            avg_score = pd.Series(
                {
                    "North": df["North Score"].mean(),
                    "South": df["South Score"].mean(),
                    "East": df["East Score"].mean(),
                    "West": df["West Score"].mean(),
                }
            )

            st.bar_chart(avg_score, height=250)

        # ----------------------------------------------------
        # 5. Green Signal Distribution
        # ----------------------------------------------------

        if "Green Direction" in df.columns:
            st.markdown(
                '<div class="section-heading">5. Green Signal Distribution</div>',
                unsafe_allow_html=True,
            )

            green = (
                df["Green Direction"]
                .value_counts()
                .reindex(DIRECTIONS, fill_value=0)
            )

            st.bar_chart(green, height=250)

        # ----------------------------------------------------
        # 6. Green Signal Duration
        # ----------------------------------------------------

        duration_column = None

        for column in [
            "Green Duration",
            "Green Time",
            "Signal Duration",
            "Duration",
        ]:
            if column in df.columns:
                duration_column = column
                break

        if duration_column is not None:
            st.markdown(
                '<div class="section-heading">6. Green Signal Duration</div>',
                unsafe_allow_html=True,
            )

            duration_counts = (
                df[duration_column]
                .dropna()
                .value_counts()
                .sort_index()
            )

            st.bar_chart(duration_counts, height=250)

        st.caption(
            "All available analytics are shown above. Graphs are displayed "
            "individually so none are hidden by a two-column layout."
        )


# ============================================================
# TRAFFIC DATA
# ============================================================

elif page == "📋 Traffic Data":

    st.title("📋 Traffic Data")

    if not CSV_PATH.exists():
        st.info("No traffic data is available yet.")
    else:
        df = pd.read_csv(CSV_PATH)

        c1, c2, c3 = st.columns(3)
        c1.metric("TOTAL RECORDS", len(df))
        c2.metric("COLUMNS", len(df.columns))
        c3.metric(
            "LATEST GREEN",
            str(df["Green Direction"].iloc[-1])
            if "Green Direction" in df.columns and len(df)
            else "—",
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=430,
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ DOWNLOAD TRAFFIC DATA",
            csv_data,
            "traffic_data.csv",
            "text/csv",
            use_container_width=True,
        )


# ============================================================
# SIGNAL INFORMATION
# ============================================================

elif page == "🚦 Signal Information":

    st.title("🚦 Adaptive Traffic Signal Control")

    st.markdown(
        """
        <div class="info-panel">
            <p>
            The system monitors traffic from four directions and rotates the
            green signal fairly. The green duration is selected from the
            detected vehicle count for that direction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("HIGH TRAFFIC", "60 seconds")
        st.caption("15 or more vehicles")

    with c2:
        st.metric("MEDIUM TRAFFIC", "45 seconds")
        st.caption("7–14 vehicles")

    with c3:
        st.metric("LOW TRAFFIC", "30 seconds")
        st.caption("0–6 vehicles")

    st.markdown(
        '<div class="section-heading">Signal Rotation</div>',
        unsafe_allow_html=True,
    )

    st.info("NORTH → SOUTH → EAST → WEST → NORTH")

    st.write(
        "Every direction gets a turn. The green duration is selected "
        "according to the detected traffic level."
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.markdown(
        """
        <div class="info-panel">
            <h3>Smart Traffic Management</h3>
            <p>
            An AI-based adaptive traffic signal management system designed
            to monitor traffic from four road directions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Project Objective</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "The system uses YOLO object detection to identify vehicles, "
        "classify traffic levels and dynamically determine green signal duration."
    )

    st.markdown(
        '<div class="section-heading">Technologies Used</div>',
        unsafe_allow_html=True,
    )

    st.write("Python • YOLO • OpenCV • Streamlit • Pandas")

    st.markdown(
        '<div class="section-heading">Vehicles Detected</div>',
        unsafe_allow_html=True,
    )

    st.write("🚗 Cars   |   🏍 Motorcycles   |   🚌 Buses   |   🚚 Trucks")

    st.markdown(
        '<div class="section-heading">Adaptive Signal Timing</div>',
        unsafe_allow_html=True,
    )

    timing_df = pd.DataFrame(
        {
            "Traffic Level": ["High", "Medium", "Low"],
            "Vehicle Count": ["15 or more", "7–14", "0–6"],
            "Green Time": ["60 seconds", "45 seconds", "30 seconds"],
        }
    )

    st.table(timing_df)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">🚦 Smart Traffic Management System • AI-Based Adaptive Traffic Analysis</div>',
    unsafe_allow_html=True,
)
