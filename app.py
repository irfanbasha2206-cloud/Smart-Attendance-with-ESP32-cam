"""
Smart Auto Attendance System
Flask + SQLite3 + face_recognition + ESP32-CAM
"""

import os, sys, threading, time, csv, json, io, base64, pickle
from datetime import datetime, date
from functools import wraps

from flask import (
    Flask, render_template, redirect, url_for, request,
    session, jsonify, Response, send_file, flash, stream_with_context
)
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import cv2
    import numpy as np
except ImportError:
    print("[ERROR] pip install opencv-python-headless numpy"); sys.exit(1)

try:
    import face_recognition
except ImportError:
    print("[ERROR] pip install face_recognition"); sys.exit(1)

try:
    import requests as req_lib
except ImportError:
    print("[ERROR] pip install requests"); sys.exit(1)

import sqlite3
from PIL import Image

# ── App setup ──────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "attend-secret-2024-xyz")

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")

# ── Global state ───────────────────────────────────────────────
_config = {
    "ESP32_IP": "192.168.1.100",
    "DETECTION_MODEL": "hog",
    "PROCESS_EVERY": 2,
    "FETCH_TIMEOUT": 3,
}
config_lock = threading.Lock()

_esp_stats = {
    "frames_processed": 0,
    "faces_detected": 0,
    "fps": 0.0,
    "esp32_online": False,
    "last_error": "",
    "face_count_current": 0,
}
stats_lock = threading.Lock()

# Real-time notification queue
_notifications = []
_notif_lock = threading.Lock()

# Known face cache
_known_faces = []        # list of {"user_id": ..., "encoding": ndarray, "name": ...}
_faces_loaded = False
_faces_lock = threading.Lock()


# ── Database ────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            face_encoding BLOB,
            face_image BLOB,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            marked_by TEXT DEFAULT 'manual',
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)

    # Default settings
    defaults = [
        ("ESP32_IP", "192.168.1.100"),
        ("DETECTION_MODEL", "hog"),
        ("PROCESS_EVERY", "2"),
        ("FETCH_TIMEOUT", "3"),
    ]
    for k, v in defaults:
        c.execute("INSERT OR IGNORE INTO settings VALUES (?,?)", (k, v))

    # Seed admin
    admin_hash = generate_password_hash("admin123")
    c.execute("""INSERT OR IGNORE INTO users (fullname, email, password_hash, role)
                 VALUES (?, ?, ?, ?)""",
              ("Administrator", "admin@attend.com", admin_hash, "admin"))
    conn.commit()
    conn.close()


def load_settings():
    global _config
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    with config_lock:
        for row in rows:
            _config[row["key"]] = row["value"]
        _config["PROCESS_EVERY"] = int(_config.get("PROCESS_EVERY", 2))
        _config["FETCH_TIMEOUT"] = int(_config.get("FETCH_TIMEOUT", 3))


def load_known_faces():
    global _known_faces, _faces_loaded
    conn = get_db()
    rows = conn.execute(
        "SELECT id, fullname, face_encoding FROM users WHERE face_encoding IS NOT NULL"
    ).fetchall()
    conn.close()
    faces = []
    for row in rows:
        try:
            enc = pickle.loads(row["face_encoding"])
            faces.append({"user_id": row["id"], "name": row["fullname"], "encoding": enc})
        except Exception:
            pass
    with _faces_lock:
        _known_faces = faces
        _faces_loaded = True


# ── Auth helpers ────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("attendance_page"))
        return f(*args, **kwargs)
    return decorated


# ── Notification helpers ────────────────────────────────────────
def push_notification(message, kind="success"):
    with _notif_lock:
        _notifications.insert(0, {
            "msg": message,
            "kind": kind,
            "ts": datetime.now().strftime("%H:%M:%S")
        })
        if len(_notifications) > 50:
            _notifications.pop()


# ── ESP32 helpers ───────────────────────────────────────────────
def get_capture_url():
    with config_lock:
        ip = _config["ESP32_IP"]
    return f"http://{ip}/capture"


def get_health_url():
    with config_lock:
        ip = _config["ESP32_IP"]
    return f"http://{ip}/health"


def fetch_frame():
    with config_lock:
        timeout = int(_config["FETCH_TIMEOUT"])
    try:
        resp = req_lib.get(get_capture_url(), timeout=timeout)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Bad JPEG")
        return img
    except Exception as e:
        with stats_lock:
            _esp_stats["last_error"] = str(e)
        return None


def detect_and_recognize(bgr_frame):
    with config_lock:
        model = _config["DETECTION_MODEL"]
    small = cv2.resize(bgr_frame, (0, 0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model=model)
    encodings = face_recognition.face_encodings(rgb, locations)
    scaled_locs = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locations]

    recognized = []
    with _faces_lock:
        known = list(_known_faces)

    for enc, loc in zip(encodings, scaled_locs):
        name = "Unknown"
        user_id = None
        if known:
            known_encs = [f["encoding"] for f in known]
            matches = face_recognition.compare_faces(known_encs, enc, tolerance=0.5)
            dists = face_recognition.face_distance(known_encs, enc)
            best = int(np.argmin(dists))
            if matches[best]:
                name = known[best]["name"]
                user_id = known[best]["user_id"]
        recognized.append({"name": name, "user_id": user_id, "loc": loc})
    return recognized


def annotate_frame(frame, recognized):
    out = frame.copy()
    for r in recognized:
        top, right, bottom, left = r["loc"]
        color = (0, 230, 90) if r["user_id"] else (60, 60, 220)
        cv2.rectangle(out, (left, top), (right, bottom), color, 2)
        label = r["name"]
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(out, (left, top - lh - 8), (left + lw + 6, top), color, -1)
        cv2.putText(out, label, (left + 3, top - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    ts = datetime.now().strftime("%H:%M:%S")
    hud = f"Faces: {len(recognized)}  |  {ts}"
    cv2.putText(out, hud, (8, 22), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 230, 90), 1, cv2.LINE_AA)
    if not recognized:
        cv2.circle(out, (out.shape[1] - 16, 16), 6, (60, 60, 220), -1)
    return out


def offline_frame():
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(blank, "ESP32 Offline", (40, 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (60, 60, 220), 1)
    cv2.putText(blank, "Check IP in Settings", (20, 140),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (120, 120, 120), 1)
    _, buf = cv2.imencode(".jpg", blank)
    return buf.tobytes()


# Auto-mark tracker: user_id -> last marked date (avoid duplicate daily marks)
_auto_marked_today = set()
_auto_marked_lock = threading.Lock()


def auto_mark_attendance(user_id, name):
    today = date.today().isoformat()
    key = (user_id, today)
    with _auto_marked_lock:
        if key in _auto_marked_today:
            return False
        _auto_marked_today.add(key)

    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM attendance WHERE user_id=? AND date=?", (user_id, today)
    ).fetchone()
    if not existing:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO attendance (user_id, timestamp, date, marked_by) VALUES (?,?,?,?)",
            (user_id, now, today, "esp32")
        )
        conn.commit()
        push_notification(f"✅ {name} marked present via ESP32-CAM at {datetime.now().strftime('%H:%M:%S')}", "success")
        conn.close()
        return True
    conn.close()
    return False


def generate_stream():
    fps_timer = time.time()
    fps_frames = 0
    frame_count = 0
    last_recognized = []

    while True:
        frame = fetch_frame()
        with config_lock:
            process_every = int(_config["PROCESS_EVERY"])

        if frame is None:
            with stats_lock:
                _esp_stats["esp32_online"] = False
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + offline_frame() + b"\r\n")
            time.sleep(0.4)
            continue

        with stats_lock:
            _esp_stats["esp32_online"] = True
            _esp_stats["last_error"] = ""

        frame_count += 1
        if frame_count % process_every == 0:
            last_recognized = detect_and_recognize(frame)
            for r in last_recognized:
                if r["user_id"]:
                    auto_mark_attendance(r["user_id"], r["name"])

        annotated = annotate_frame(frame, last_recognized)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])

        fps_frames += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            with stats_lock:
                _esp_stats["fps"] = round(fps_frames / elapsed, 1)
                _esp_stats["frames_processed"] += fps_frames
                _esp_stats["faces_detected"] += len(last_recognized)
                _esp_stats["face_count_current"] = len(last_recognized)
            fps_frames = 0
            fps_timer = time.time()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")


# ── Routes: Auth ────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("attendance_page"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("attendance_page"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["fullname"] = user["fullname"]
            session["email"] = user["email"]
            session["role"] = user["role"]
            flash(f"Welcome back, {user['fullname']}!", "success")
            return redirect(url_for("dashboard" if user["role"] == "admin" else "attendance_page"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("attendance_page"))
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        face_data = request.form.get("face_data", "")

        if not all([fullname, email, password, confirm]):
            flash("All fields are required.", "danger")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")

        conn = get_db()
        existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            conn.close()
            flash("Email already registered.", "danger")
            return render_template("register.html")

        face_encoding_blob = None
        face_image_blob = None

        if face_data:
            try:
                header, b64data = face_data.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb)
                encs = face_recognition.face_encodings(rgb, locs)
                if encs:
                    face_encoding_blob = pickle.dumps(encs[0])
                    face_image_blob = img_bytes
                else:
                    flash("No face detected in the captured image. Please retake.", "warning")
                    return render_template("register.html")
            except Exception as e:
                flash(f"Face processing error: {e}", "danger")
                return render_template("register.html")

        pw_hash = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (fullname, email, password_hash, face_encoding, face_image) VALUES (?,?,?,?,?)",
            (fullname, email, pw_hash, face_encoding_blob, face_image_blob)
        )
        conn.commit()
        conn.close()

        # Reload face cache
        load_known_faces()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ── Routes: Attendance ──────────────────────────────────────────
@app.route("/attendance")
@login_required
def attendance_page():
    today = date.today().isoformat()
    conn = get_db()
    user_id = session["user_id"]

    today_record = conn.execute(
        "SELECT * FROM attendance WHERE user_id=? AND date=? ORDER BY timestamp DESC LIMIT 1",
        (user_id, today)
    ).fetchone()

    # Attendance stats
    total_days = conn.execute(
        "SELECT COUNT(DISTINCT date) FROM attendance WHERE user_id=?", (user_id,)
    ).fetchone()[0]

    # Working days (approximate: count distinct dates with any attendance)
    all_days = conn.execute("SELECT COUNT(DISTINCT date) FROM attendance").fetchone()[0]
    percentage = round((total_days / all_days * 100) if all_days > 0 else 0, 1)

    conn.close()
    with config_lock:
        esp32_ip = _config["ESP32_IP"]
    return render_template("attendance.html",
                           today=today,
                           today_record=today_record,
                           total_days=total_days,
                           percentage=percentage,
                           esp32_ip=esp32_ip)


@app.route("/attendance/mark", methods=["POST"])
@login_required
def mark_attendance():
    today = date.today().isoformat()
    user_id = session["user_id"]
    name = session["fullname"]
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM attendance WHERE user_id=? AND date=?", (user_id, today)
    ).fetchone()
    if existing:
        conn.close()
        return jsonify({"ok": False, "msg": "Already marked for today."})
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO attendance (user_id, timestamp, date, marked_by) VALUES (?,?,?,?)",
        (user_id, now, today, "manual")
    )
    conn.commit()
    conn.close()
    push_notification(f"✅ {name} marked present manually at {datetime.now().strftime('%H:%M:%S')}", "success")
    return jsonify({"ok": True, "msg": f"Attendance marked at {now}", "time": now})


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(
        stream_with_context(generate_stream()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/esp32_status")
@login_required
def esp32_status():
    try:
        r = req_lib.get(get_health_url(), timeout=2)
        data = r.json()
        data["reachable"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"reachable": False, "error": str(e)})


@app.route("/esp_stats")
@login_required
def esp_stats():
    with stats_lock:
        return jsonify(dict(_esp_stats))


# ── Routes: History ─────────────────────────────────────────────
@app.route("/history")
@login_required
def history():
    return render_template("history.html")


@app.route("/api/history")
@login_required
def api_history():
    user_id = session["user_id"]
    role = session["role"]
    page = int(request.args.get("page", 1))
    per_page = 20
    offset = (page - 1) * per_page
    search_user = request.args.get("user", "").strip()
    search_date = request.args.get("date", "").strip()

    conn = get_db()
    if role == "admin":
        base_q = """
            SELECT a.id, u.fullname, u.email, a.timestamp, a.date, a.marked_by
            FROM attendance a JOIN users u ON a.user_id=u.id
            WHERE 1=1
        """
        params = []
        if search_user:
            base_q += " AND (u.fullname LIKE ? OR u.email LIKE ?)"
            params += [f"%{search_user}%", f"%{search_user}%"]
        if search_date:
            base_q += " AND a.date=?"
            params.append(search_date)
    else:
        base_q = """
            SELECT a.id, u.fullname, u.email, a.timestamp, a.date, a.marked_by
            FROM attendance a JOIN users u ON a.user_id=u.id
            WHERE a.user_id=?
        """
        params = [user_id]
        if search_date:
            base_q += " AND a.date=?"
            params.append(search_date)

    total = conn.execute(f"SELECT COUNT(*) FROM ({base_q})", params).fetchone()[0]
    rows = conn.execute(base_q + " ORDER BY a.timestamp DESC LIMIT ? OFFSET ?",
                        params + [per_page, offset]).fetchall()
    conn.close()

    records = [dict(r) for r in rows]
    return jsonify({
        "records": records,
        "total": total,
        "page": page,
        "pages": (total + per_page - 1) // per_page
    })


# ── Routes: Dashboard (Admin) ───────────────────────────────────
@app.route("/dashboard")
@admin_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/dashboard_stats")
@admin_required
def dashboard_stats():
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0]
    today = date.today().isoformat()
    present_today = conn.execute(
        "SELECT COUNT(DISTINCT user_id) FROM attendance WHERE date=?", (today,)
    ).fetchone()[0]
    total_records = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]

    # Per-user attendance percentage
    users = conn.execute("SELECT id, fullname, email FROM users WHERE role='user'").fetchall()
    all_dates = conn.execute("SELECT COUNT(DISTINCT date) FROM attendance").fetchone()[0] or 1

    user_stats = []
    for u in users:
        days = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM attendance WHERE user_id=?", (u["id"],)
        ).fetchone()[0]
        pct = round(days / all_dates * 100, 1)
        last = conn.execute(
            "SELECT timestamp FROM attendance WHERE user_id=? ORDER BY timestamp DESC LIMIT 1",
            (u["id"],)
        ).fetchone()
        user_stats.append({
            "id": u["id"],
            "fullname": u["fullname"],
            "email": u["email"],
            "days_present": days,
            "percentage": pct,
            "last_seen": last["timestamp"] if last else "Never"
        })

    # Weekly attendance count (last 7 days)
    weekly = conn.execute("""
        SELECT date, COUNT(*) as cnt
        FROM attendance
        WHERE date >= date('now', '-6 days')
        GROUP BY date ORDER BY date
    """).fetchall()

    conn.close()
    return jsonify({
        "total_users": total_users,
        "present_today": present_today,
        "absent_today": total_users - present_today,
        "total_records": total_records,
        "user_stats": user_stats,
        "weekly": [{"date": r["date"], "count": r["cnt"]} for r in weekly]
    })


# ── Routes: Settings ────────────────────────────────────────────
@app.route("/settings", methods=["GET", "POST"])
@admin_required
def settings():
    if request.method == "POST":
        new_ip = request.form.get("esp32_ip", "").strip()
        new_model = request.form.get("detection_model", "hog")
        new_process = request.form.get("process_every", "2")
        new_timeout = request.form.get("fetch_timeout", "3")

        conn = get_db()
        updates = [
            ("ESP32_IP", new_ip),
            ("DETECTION_MODEL", new_model),
            ("PROCESS_EVERY", new_process),
            ("FETCH_TIMEOUT", new_timeout),
        ]
        for k, v in updates:
            conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (k, v))
        conn.commit()
        conn.close()
        load_settings()
        flash("Settings saved successfully.", "success")
        return redirect(url_for("settings"))

    with config_lock:
        cfg = dict(_config)
    return render_template("settings.html", config=cfg)


# ── Routes: Export CSV ──────────────────────────────────────────
@app.route("/export_csv")
@admin_required
def export_csv():
    search_date = request.args.get("date", "")
    search_user = request.args.get("user", "")

    conn = get_db()
    q = """
        SELECT u.fullname, u.email, a.date, a.timestamp, a.marked_by
        FROM attendance a JOIN users u ON a.user_id=u.id
        WHERE 1=1
    """
    params = []
    if search_date:
        q += " AND a.date=?"; params.append(search_date)
    if search_user:
        q += " AND (u.fullname LIKE ? OR u.email LIKE ?)"
        params += [f"%{search_user}%", f"%{search_user}%"]
    q += " ORDER BY a.timestamp DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Full Name", "Email", "Date", "Timestamp", "Marked By"])
    for r in rows:
        writer.writerow([r["fullname"], r["email"], r["date"], r["timestamp"], r["marked_by"]])
    output.seek(0)

    filename = f"attendance_{date.today().isoformat()}.csv"
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


# ── Routes: Notifications (SSE) ────────────────────────────────
@app.route("/api/notifications")
@login_required
def notifications_stream():
    def event_stream():
        last_count = 0
        while True:
            with _notif_lock:
                current = list(_notifications)
            if len(current) > last_count:
                new_items = current[:len(current) - last_count]
                for item in reversed(new_items):
                    yield f"data: {json.dumps(item)}\n\n"
                last_count = len(current)
            time.sleep(1)
    return Response(stream_with_context(event_stream()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/recent_notifications")
@login_required
def recent_notifications():
    with _notif_lock:
        return jsonify(_notifications[:10])


# ── Routes: Admin: User management ─────────────────────────────
@app.route("/api/users")
@admin_required
def api_users():
    conn = get_db()
    users = conn.execute(
        "SELECT id, fullname, email, role, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])


@app.route("/api/users/<int:uid>", methods=["DELETE"])
@admin_required
def delete_user(uid):
    if uid == session["user_id"]:
        return jsonify({"ok": False, "msg": "Cannot delete yourself."})
    conn = get_db()
    conn.execute("DELETE FROM attendance WHERE user_id=?", (uid,))
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    load_known_faces()
    return jsonify({"ok": True})


# ── Routes: Face image ──────────────────────────────────────────
@app.route("/api/face_image/<int:uid>")
@login_required
def face_image(uid):
    if session["role"] != "admin" and session["user_id"] != uid:
        return "", 403
    conn = get_db()
    row = conn.execute("SELECT face_image FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if row and row["face_image"]:
        return Response(row["face_image"], mimetype="image/jpeg")
    return "", 404


# ── Routes: Webcam face check (capture during register) ─────────
@app.route("/api/check_face", methods=["POST"])
def check_face():
    data = request.json.get("image", "")
    if not data:
        return jsonify({"ok": False, "msg": "No image"})
    try:
        header, b64 = data.split(",", 1)
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        return jsonify({"ok": True, "count": len(locs), "detected": len(locs) > 0})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


# ── Error handlers ──────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("login.html"), 404


@app.errorhandler(403)
def forbidden(e):
    flash("Access denied.", "danger")
    return redirect(url_for("attendance_page"))


# ── Startup ─────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    load_settings()
    load_known_faces()
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print("  Smart Auto Attendance System")
    print(f"  http://localhost:{port}")
    print("  Admin: admin@attend.com / admin123")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
