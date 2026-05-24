
import board, busio, adafruit_mlx90640
import onnxruntime as ort
import numpy as np, cv2, serial, pynmea2
import json, time, os, sys
from datetime import datetime
from scipy.ndimage import zoom

MODEL_PATH     = "vit_solar_fault_quantized.onnx"
IMG_SIZE       = 224
CONFIDENCE_THR = 0.70
LOG_FILE       = "fault_log.json"
CAPTURES_DIR   = "captures"
GPS_PORT       = "/dev/serial0"
GPS_BAUD       = 9600
SAVE_IMAGES    = True
CLASS_NAMES    = ["Cell-Fault","Cracking","Diode-Fault","Hot-Spot",
                  "No-Anomaly","Offline-Module","Shadowing","Soiling","Vegetation"]
NORMAL_CLASS   = "No-Anomaly"
PANEL_TEMP_MIN     = 25.0
PANEL_TEMP_MAX     = 85.0
PANEL_MIN_COVERAGE = 0.50
PANEL_MIN_SPREAD   = 5.0

def banner(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

class PanelDetector:
    def __init__(self):
        self.temp_min     = PANEL_TEMP_MIN
        self.temp_max     = PANEL_TEMP_MAX
        self.min_coverage = PANEL_MIN_COVERAGE
        self.min_spread   = PANEL_MIN_SPREAD

    def is_panel_in_view(self, temp_array):
        spread = float(temp_array.max() - temp_array.min())
        if spread < self.min_spread:
            return False, f"no thermal variation (spread={spread:.1f}C)", 0.0
        in_range = np.sum((temp_array >= self.temp_min) & (temp_array <= self.temp_max))
        coverage = float(in_range) / temp_array.size
        if coverage < self.min_coverage:
            return False, f"coverage too low ({coverage:.0%}) — point camera at panel", coverage
        return True, f"panel detected ({coverage:.0%} coverage)", coverage

class ThermalCamera:
    def __init__(self):
        print("[Camera] Connecting to MLX90640...")
        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            self.cam = adafruit_mlx90640.MLX90640(i2c)
            self.cam.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
            self._buffer = [0] * 768
            print("[Camera] Warming up...")
            for _ in range(3):
                try: self.cam.getFrame(self._buffer)
                except: pass
                time.sleep(0.6)
            print(f"[Camera] Connected!")
        except Exception as e:
            print(f"[Camera] FAILED: {e}")
            print("  Check: VCC->Pin1(3.3V) GND->Pin6 SDA->Pin3 SCL->Pin5")
            sys.exit(1)

    def read(self):
        self.cam.getFrame(self._buffer)
        return np.array(self._buffer).reshape(24, 32)

class Preprocessor:
    def process(self, temp_array):
        t_min, t_max = temp_array.min(), temp_array.max()
        norm     = ((temp_array - t_min) / max(t_max - t_min, 1e-6) * 255).astype(np.uint8)
        upscaled = zoom(norm, (IMG_SIZE/24, IMG_SIZE/32), order=3)
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
        rgb      = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)
        img      = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor   = np.expand_dims(np.transpose(img, (2,0,1)), axis=0).astype(np.float32)
        return tensor, upscaled

class FaultClassifier:
    def __init__(self):
        print(f"[Model] Loading {MODEL_PATH}...")
        self.session      = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        self.input_name   = self.session.get_inputs()[0].name
        self.output_name  = self.session.get_outputs()[0].name
        self.preprocessor = Preprocessor()
        print(f"[Model] Loaded!")

    def predict(self, temp_array):
        tensor, upscaled = self.preprocessor.process(temp_array)
        logits  = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        exp_l   = np.exp(logits - np.max(logits))
        probs   = exp_l / exp_l.sum()
        idx     = int(np.argmax(probs))
        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]),4) for i in range(len(CLASS_NAMES))}
        return CLASS_NAMES[idx], float(probs[idx]), prob_dict, upscaled

class GPSReader:
    def __init__(self):
        print(f"[GPS] Connecting on {GPS_PORT}...")
        try:
            self.serial  = serial.Serial(GPS_PORT, baudrate=GPS_BAUD, timeout=1)
            self.enabled = True
            print("[GPS] Connected!")
        except Exception as e:
            self.enabled = False
            print(f"[GPS] Not available: {e}")

    def read(self):
        if not self.enabled:
            return {"lat": None, "lon": None}
        try:
            line = self.serial.readline().decode("ascii", errors="replace").strip()
            if line.startswith(("$GPRMC","$GNRMC")):
                msg = pynmea2.parse(line)
                if msg.status == "A":
                    return {"lat": round(msg.latitude,6), "lon": round(msg.longitude,6)}
        except: pass
        return {"lat": None, "lon": None}

class FaultLogger:
    def __init__(self):
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        print(f"[Logger] Logging to {LOG_FILE}")

    def log(self, label, confidence, prob_dict, temp_array, gps, upscaled):
        ts  = datetime.now()
        rec = {
            "timestamp":  ts.isoformat(),
            "fault":      label,
            "confidence": round(confidence, 4),
            "all_scores": prob_dict,
            "thermal": {
                "min_temp_c":  round(float(temp_array.min()), 2),
                "max_temp_c":  round(float(temp_array.max()), 2),
                "mean_temp_c": round(float(temp_array.mean()), 2),
                "delta_temp":  round(float(temp_array.max()-temp_array.min()), 2),
            },
            "gps":   gps,
            "image": None
        }
        if SAVE_IMAGES:
            path = os.path.join(CAPTURES_DIR, f"{label}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path, cv2.applyColorMap(upscaled, cv2.COLORMAP_INFERNO))
            rec["image"] = path
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

def main():
    banner("Solar Panel Fault Detection — Starting Up")
    detector   = PanelDetector()
    camera     = ThermalCamera()
    classifier = FaultClassifier()
    gps        = GPSReader()
    logger     = FaultLogger()
    banner("Running — point camera at solar panel  (Ctrl+C to stop)")

    frame_count = fault_count = 0
    last_msg    = ""
    start_time  = datetime.now()

    try:
        while True:
            try:
                temp_array = camera.read()
            except Exception as e:
                print(f"[Camera] Read error: {e}")
                time.sleep(1.0)
                continue

            frame_count += 1
            panel_found, reason, coverage = detector.is_panel_in_view(temp_array)

            if not panel_found:
                msg = f"[{datetime.now().strftime("%H:%M:%S")}] NO PANEL | {reason}"
                if msg != last_msg:
                    print(msg)
                    last_msg = msg
                time.sleep(0.55)
                continue

            last_msg = ""
            label, confidence, prob_dict, upscaled = classifier.predict(temp_array)
            location = gps.read()
            is_fault = (label != NORMAL_CLASS and confidence >= CONFIDENCE_THR)
            flag     = "FAULT" if is_fault else "OK"
            gps_str  = f"{location['lat']:.5f},{location['lon']:.5f}" if location["lat"] else "waiting..."
            print(f"[{datetime.now().strftime("%H:%M:%S")}] {flag} | {label:<16} {confidence:5.1%} | "
                  f"Panel:{coverage:.0%} | Temp:{temp_array.min():.0f}-{temp_array.max():.0f}C | "
                  f"GPS:{gps_str} | Faults:{fault_count}")

            if is_fault:
                fault_count += 1
                rec = logger.log(label, confidence, prob_dict, temp_array, location, upscaled)
                print(f"  >>> Logged fault #{fault_count}: {rec['fault']} @ delta{rec['thermal']['delta_temp']}C")

            time.sleep(0.55)

    except KeyboardInterrupt:
        elapsed = datetime.now() - start_time
        banner("Stopped")
        print(f"  Runtime : {str(elapsed).split('.')[0]}")
        print(f"  Frames  : {frame_count}")
        print(f"  Faults  : {fault_count}")
        print(f"  Log     : {LOG_FILE}")

if __name__ == "__main__":
    main()
