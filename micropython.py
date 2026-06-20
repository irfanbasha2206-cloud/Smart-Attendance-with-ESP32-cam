# ─────────────────────────────────────────────────────────────
#  ESP32-CAM MicroPython firmware  –  upload via Thonny
#  Endpoints:
#    GET /capture  → single JPEG frame
#    GET /health   → JSON status check
# ─────────────────────────────────────────────────────────────
import camera
import network
import socket
import time
import machine

# ── Reduce CPU to 80 MHz to prevent brownout during camera init ──
# Default 240 MHz draws too much current on USB power → triggers reset.
machine.freq(80_000_000)
print("[Power] CPU set to 80 MHz (prevents brownout)")

# ── Wi-Fi credentials ──────────────────────────────────────
SSID     = "Irfan AI"
PASSWORD = "11223344"

# ── Connect to Wi-Fi ───────────────────────────────────────
print("\n[WiFi] Connecting to", SSID, "...")
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

attempts = 0
while not wlan.isconnected():
    time.sleep(0.5)
    attempts += 1
    if attempts > 40:          # 20-second timeout
        print("[WiFi] FAILED to connect. Rebooting...")
        import machine
        machine.reset()

ip = wlan.ifconfig()[0]
print("[WiFi] Connected!")
print("[WiFi] ESP32 IP address :", ip)
print("[WiFi] Open in browser  : http://" + ip + "/capture")
print()

# ── Init camera ────────────────────────────────────────────
try:
    camera.deinit()
except:
    pass

# camera.init() signature varies across MicroPython ESP32-CAM firmware builds.
# Try each form in order — the first one that works is used.
_cam_ok = False
for _init_fn in [
    lambda: camera.init(0, format=camera.JPEG, framesize=camera.FRAME_VGA, quality=12),
    lambda: camera.init(0, format=camera.JPEG, framesize=camera.FRAME_VGA),
    lambda: camera.init(0, format=camera.JPEG),
    lambda: camera.init(0),
    lambda: camera.init(),
]:
    try:
        _init_fn()
        _cam_ok = True
        break
    except (TypeError, AttributeError, Exception):
        pass

if not _cam_ok:
    print("[Camera] ERROR: could not initialise camera — check firmware")
else:
    # Optional quality/resolution setters (may not exist on all builds)
    for _fn, _label in [
        (lambda: camera.framesize(camera.FRAME_VGA), "framesize VGA"),
        (lambda: camera.quality(12),                  "quality 12"),
    ]:
        try:
            _fn()
        except:
            pass
    print("[Camera] Initialized OK")

# ── HTTP helpers ───────────────────────────────────────────

def read_request(client):
    """Read HTTP request line, return (method, path) or (None, None)."""
    try:
        raw   = client.recv(1024).decode('utf-8', 'ignore')
        first = raw.split('\r\n')[0]
        parts = first.split(' ')
        if len(parts) >= 2:
            return parts[0], parts[1].split('?')[0]
    except:
        pass
    return None, None


def send_headers(client, status, content_type, extra_headers=b''):
    client.send(('HTTP/1.1 ' + status + '\r\n').encode())
    client.send(('Content-Type: ' + content_type + '\r\n').encode())
    client.send(b'Access-Control-Allow-Origin: *\r\n')
    client.send(b'Cache-Control: no-cache\r\n')
    if extra_headers:
        client.send(extra_headers)
    client.send(b'\r\n')


def handle_capture(client):
    """Capture one JPEG frame and send it."""
    buf = camera.capture()
    if buf:
        send_headers(
            client,
            '200 OK',
            'image/jpeg',
            ('Content-Length: ' + str(len(buf)) + '\r\n').encode()
        )
        # Send in 1 KB chunks to avoid memory issues
        view = memoryview(buf)
        for i in range(0, len(buf), 1024):
            client.send(view[i:i + 1024])
    else:
        send_headers(client, '500 Internal Server Error', 'text/plain')
        client.send(b'Camera capture failed')


def handle_health(client):
    """Simple JSON health check so the web app can verify connectivity."""
    body = b'{"status":"ok","device":"ESP32-CAM"}'
    send_headers(
        client,
        '200 OK',
        'application/json',
        ('Content-Length: ' + str(len(body)) + '\r\n').encode()
    )
    client.send(body)


def handle_not_found(client):
    send_headers(client, '404 Not Found', 'text/plain')
    client.send(b'Not found. Use /capture or /health')


# ── Main server loop ───────────────────────────────────────
addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
srv  = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(addr)
srv.listen(3)

print("[Server] Listening on port 80")
print("[Server] Waiting for requests...\n")

req_count = 0
while True:
    cl = None
    try:
        cl, remote = srv.accept()
        cl.settimeout(5)          # 5-second per-request timeout
        method, path = read_request(cl)

        if method is None:
            pass
        elif path == '/capture':
            handle_capture(cl)
            req_count += 1
            if req_count % 10 == 0:
                print("[Server] Frames served:", req_count)
        elif path == '/health':
            handle_health(cl)
        else:
            handle_not_found(cl)

    except OSError as e:
        print("[Server] Socket error:", e)
    except Exception as e:
        print("[Server] Error:", e)
    finally:
        if cl:
            try:
                cl.close()
            except:
                pass

