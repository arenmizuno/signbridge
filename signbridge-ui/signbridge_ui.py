# ============================================================
# ui-5.py — SignBridge  (No LLM · Top-5 Display · TTS Read-out)
# Setup:
#   conda activate signbridge_demo
#   pip install mediapipe==0.10.9 opencv-python onnxruntime numpy
#   python ui-5.py
#
# Changes from ui-4.py:
#   - Removed OpenAI/GPT dependency entirely
#   - Added macOS `say` command for TTS (background thread, MacBook speakers)
#   - Top-5 panel already existed — kept and polished
#   - Thumbs-up now clears the word stream instead of sending to GPT
#   - Removed chat log (replaced by spoken output)
# ============================================================
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import json
import os
import threading
import subprocess

# ── TTS via macOS `say` command ───────────────────────────────
TTS_DEVICE = "MacBook Pro Speakers"   # change if you use a different output

def speak(text: str):
    """Speak `text` in a background thread so the video loop is not blocked."""
    def _run():
        subprocess.run(['say', '-a', TTS_DEVICE, text])
    t = threading.Thread(target=_run, daemon=True)
    t.start()

# ── Config ────────────────────────────────────────────────────
DATA_DIR       = './model'
ONNX_PATH      = os.path.join(DATA_DIR, 'signbridge_v3.onnx')

CONF_THRESHOLD = 0.05   # minimum confidence to accept a sign

# ── Landmark selection — exact match to preprocessing ─────────
LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
NOSE = [1, 2, 98, 327]
REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

FACE_LANDMARKS_SORTED = sorted(set(LIP + NOSE + REYE + LEYE))

N_FACE   = len(FACE_LANDMARKS_SORTED)   # 76
N_HAND   = 21
N_LMARKS = N_FACE + N_HAND + N_HAND     # 118
N_POS    = N_LMARKS * 3                 # 354
N_TOTAL  = N_POS * 2                    # 708 (position + velocity)
RH_START = N_FACE + N_HAND

print(f"Face landmarks : {N_FACE}")
print(f"Total landmarks: {N_LMARKS}")
print(f"Position feats : {N_POS}")
print(f"Total features : {N_TOTAL}")
assert N_TOTAL == 708, f"Expected 708, got {N_TOTAL}"

# ── Load ONNX model ───────────────────────────────────────────
print("Loading model...")
sess        = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
INPUT_X     = sess.get_inputs()[0].name
INPUT_LEN   = sess.get_inputs()[1].name if len(sess.get_inputs()) > 1 else None
OUTPUT_NAME = sess.get_outputs()[0].name
print(f"Model loaded ✓  inputs: {[i.name for i in sess.get_inputs()]}")

# ── Load label map ────────────────────────────────────────────
with open(os.path.join(DATA_DIR, 'full250_label_map.json')) as f:
    raw_map = json.load(f)

if 'idx_to_sign' in raw_map:
    idx_to_sign = {int(k): v for k, v in raw_map['idx_to_sign'].items()}
elif 'label_to_word' in raw_map:
    idx_to_sign = {int(k): v for k, v in raw_map['label_to_word'].items()}
elif 'word_to_label' in raw_map:
    idx_to_sign = {int(v): k for k, v in raw_map['word_to_label'].items()}
else:
    idx_to_sign = {}
    for k, v in raw_map.items():
        try:
            idx_to_sign[int(k)] = v
        except (ValueError, TypeError):
            pass

print(f"Labels loaded ✓  {len(idx_to_sign)} signs")
print(f"Sample: {dict(list(idx_to_sign.items())[:3])}")

# ── MediaPipe Holistic ────────────────────────────────────────
print("Loading MediaPipe Holistic...")
mp_holistic = mp.solutions.holistic
holistic    = mp_holistic.Holistic(
    min_detection_confidence = 0.5,
    min_tracking_confidence  = 0.5,
    model_complexity         = 1,
)
print("MediaPipe Holistic ready ✓")

# ── Extractor ─────────────────────────────────────────────────
class Extractor:
    def __init__(self):
        self.history = []

    def add_frame(self, frame_rgb):
        results = holistic.process(frame_rgb)
        lm      = np.full((N_LMARKS, 3), np.nan, dtype=np.float32)

        if results.face_landmarks:
            face = results.face_landmarks.landmark
            for i, fidx in enumerate(FACE_LANDMARKS_SORTED):
                lm[i, 0] = face[fidx].x
                lm[i, 1] = face[fidx].y
                lm[i, 2] = face[fidx].z

        if results.left_hand_landmarks:
            for i, hlm in enumerate(results.left_hand_landmarks.landmark):
                lm[N_FACE + i, 0] = hlm.x
                lm[N_FACE + i, 1] = hlm.y
                lm[N_FACE + i, 2] = hlm.z

        if results.right_hand_landmarks:
            for i, hlm in enumerate(results.right_hand_landmarks.landmark):
                lm[RH_START + i, 0] = hlm.x
                lm[RH_START + i, 1] = hlm.y
                lm[RH_START + i, 2] = hlm.z

        rh       = lm[RH_START : RH_START + N_HAND, :]
        rh_wrist = rh[0, :].copy()
        if not np.isnan(rh_wrist).any():
            lm[RH_START : RH_START + N_HAND, :] -= rh_wrist
            mid   = lm[RH_START + 12, :]
            scale = np.linalg.norm(mid)
            if scale > 1e-6:
                lm[RH_START : RH_START + N_HAND, :] /= scale

        feat = np.concatenate([lm[:, 0], lm[:, 1], lm[:, 2]])
        feat = np.where(np.isnan(feat), 0.0, feat)
        self.history.append(feat)
        return results

    def build_window(self):
        if not self.history:
            return np.zeros((96, N_TOTAL), dtype=np.float32), 0
        seq          = np.array(self.history, dtype=np.float32)
        T            = len(seq)
        velocity     = np.zeros_like(seq)
        velocity[1:] = seq[1:] - seq[:-1]
        full         = np.concatenate([seq, velocity], axis=1)
        n            = min(T, 96)
        window       = np.zeros((96, N_TOTAL), dtype=np.float32)
        window[:n]   = full[:n]
        return window.astype(np.float32), n

    def reset(self):
        self.history = []


# ── Sign Capture ──────────────────────────────────────────────
class SignCapture:
    def __init__(self, max_frames=96, min_frames=10, no_hand_threshold=10):
        self.max_frames        = max_frames
        self.min_frames        = min_frames
        self.no_hand_threshold = no_hand_threshold
        self.frame_count       = 0
        self.capturing         = False
        self.no_hand_count     = 0
        self.cooldown          = False

    def update(self, hp):
        if self.cooldown:
            if not hp:
                self.cooldown = False
            return False

        if hp:
            self.no_hand_count = 0
            if not self.capturing:
                self.capturing   = True
                self.frame_count = 0
                print("  [capture] Sign started")
            self.frame_count += 1
            if self.frame_count >= self.max_frames:
                print("  [capture] Max frames reached — inferring")
                self.capturing = False
                self.cooldown  = True
                return True
        else:
            if self.capturing:
                self.no_hand_count += 1
                self.frame_count   += 1
                if self.no_hand_count >= self.no_hand_threshold:
                    if self.frame_count >= self.min_frames:
                        print(f"  [capture] Sign ended — {self.frame_count} frames")
                        self.capturing = False
                        return True
                    else:
                        print("  [capture] Too short — ignored")
                        self.reset()
        return False

    def reset(self):
        self.frame_count   = 0
        self.capturing     = False
        self.no_hand_count = 0
        self.cooldown      = False


# ── Inference ─────────────────────────────────────────────────
def run_inference(window, length):
    inputs = {INPUT_X: window[np.newaxis].astype(np.float32)}
    if INPUT_LEN:
        inputs[INPUT_LEN] = np.array([length], dtype=np.int64)
    logits = sess.run([OUTPUT_NAME], inputs)[0][0]
    probs  = np.exp(logits - logits.max())
    probs /= probs.sum()
    top5   = probs.argsort()[-5:][::-1]
    signs  = [idx_to_sign.get(int(i), str(i)) for i in top5]
    confs  = [float(probs[i]) for i in top5]
    return signs, confs


def is_thumbs_up(results):
    for hand_lms in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_lms:
            lm = hand_lms.landmark
            if (lm[4].y < lm[2].y - 0.05 and
                lm[8].y  > lm[6].y  and
                lm[12].y > lm[10].y and
                lm[16].y > lm[14].y and
                lm[20].y > lm[18].y):
                return True
    return False


def hp_from_results(results):
    return (results.left_hand_landmarks  is not None or
            results.right_hand_landmarks is not None)


# ── Draw hand landmarks ───────────────────────────────────────
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

def draw_hands(frame, results, w, h):
    for hand_lms in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_lms:
            for p in hand_lms.landmark:
                cx = int((1 - p.x) * w)
                cy = int(p.y * h)
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
            for s, e in CONN:
                lms = hand_lms.landmark
                cv2.line(frame,
                         (int((1-lms[s].x)*w), int(lms[s].y*h)),
                         (int((1-lms[e].x)*w), int(lms[e].y*h)),
                         (0,200,0), 2)


# ── Main ──────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    extractor    = Extractor()
    sign_cap     = SignCapture(max_frames=96, min_frames=10, no_hand_threshold=10)
    words        = []          # accepted top-1 words in this session
    confs        = []
    top1         = ''
    conf         = 0.0
    top5_display = []          # [(sign, prob), ...]

    print("\n=== SignBridge Ready (TTS mode) ===")
    print("Sign a word → lower hand → wait → top-5 shown, top-1 spoken")
    print("Thumbs up = clear word stream  |  Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_display     = cv2.flip(frame, 1)
        rgb_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w              = frame.shape[:2]

        results = extractor.add_frame(rgb_for_mediapipe)
        hp      = hp_from_results(results)

        sign_complete = sign_cap.update(hp)

        # ── Status bar (top) ──────────────────────────────────
        if not hp:
            cv2.putText(frame_display, 'Show your hand to sign', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        elif is_thumbs_up(results):
            # Thumbs up → clear the word stream
            words.clear()
            confs.clear()
            top5_display = []
            top1         = ''
            conf         = 0.0
            extractor.reset()
            sign_cap.reset()
            cv2.putText(frame_display, 'CLEARED ✓', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        else:
            draw_hands(frame_display, results, w, h)
            if sign_cap.capturing:
                n = sign_cap.frame_count
                cv2.putText(frame_display,
                            f'Capturing... {n}/96', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            else:
                label = f'{top1}  {conf:.0%}' if top1 else 'Ready — sign a word'
                cv2.putText(frame_display, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        # ── Run inference when sign is complete ───────────────
        if sign_complete:
            window_arr, length = extractor.build_window()
            signs, probas      = run_inference(window_arr, length)
            top1, conf         = signs[0], probas[0]
            top5_display       = list(zip(signs, probas))

            print(f"Prediction : {top1} ({conf:.0%})")
            print(f"  Top 5   : {[(s, f'{c:.0%}') for s,c in top5_display]}")

            # ── Speak the top-1 word ───────────────────────────
            if conf >= CONF_THRESHOLD:
                speak(top1)
                if not words or words[-1] != top1:
                    words.append(top1)
                    confs.append(conf)

            extractor.reset()

        # ── Top-5 panel ───────────────────────────────────────
        if top5_display:
            panel_x1, panel_y1 = 0,   75
            panel_x2, panel_y2 = 360, 255
            overlay = frame_display.copy()
            cv2.rectangle(overlay, (panel_x1, panel_y1),
                          (panel_x2, panel_y2), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame_display, 0.25, 0, frame_display)

            cv2.putText(frame_display, 'Top 5 Predictions', (12, 97),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)

            for rank, (sign, prob) in enumerate(top5_display[:5]):
                y     = 120 + rank * 26
                color = (0, 255, 120) if rank == 0 else (130, 130, 130)
                bar_w = int(prob * 200)
                cv2.rectangle(frame_display,
                              (10, y - 14), (10 + bar_w, y + 4),
                              (0, 80, 40) if rank == 0 else (40, 40, 40), -1)
                label = f'{rank+1}. {sign:<18s} {prob:5.1%}'
                cv2.putText(frame_display, label, (14, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        # ── Word stream bar (bottom) ───────────────────────────
        bar_y = h - 70
        cv2.rectangle(frame_display, (0, bar_y - 15), (w, h - 40),
                      (30, 30, 30), -1)
        x_off = 10
        for word, c in zip(words, confs):
            col = (0,220,0) if c >= 0.3 else (0,165,255) if c >= 0.1 else (0,80,255)
            cv2.putText(frame_display, word, (x_off, bar_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            x_off += len(word) * 16 + 12

        # ── Instructions footer ───────────────────────────────
        cv2.putText(frame_display,
                    'Sign → lower hand → wait  |  Thumbs up = clear  |  Q = quit',
                    (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        cv2.imshow('SignBridge', frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    holistic.close()
    cap.release()
    cv2.destroyAllWindows()
    print("SignBridge closed.")


if __name__ == '__main__':
    main()