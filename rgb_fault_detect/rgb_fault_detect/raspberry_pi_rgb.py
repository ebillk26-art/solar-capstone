#!/usr/bin/env python3
"""
RGB Solar Panel Defect Detector - Raspberry Pi
Model: yolov11m (98.5% accuracy)
Classes: Bird-drop, Clean, Dusty, Physical-Damage
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np

# Configuration
MODEL_PATH = 'best_rgb_4class.pt'  # or best.onnx for faster inference
CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Physical-Damage']

# Load model
print("Loading model...")
model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")
print(f"Classes: {len(CLASSES)}\n")

def detect_defect(image_path):
    """
    Detect defect in solar panel image

    Returns:
        class_name: Detected class
        confidence: Prediction confidence (0-1)
        all_probs: Probabilities for all classes
    """
    results = model(image_path, verbose=False)
    probs = results[0].probs

    class_id = probs.top1
    confidence = probs.top1conf.item()
    class_name = CLASSES[class_id]
    all_probs = probs.data.cpu().numpy()

    return class_name, confidence, all_probs

def print_detection(class_name, confidence, all_probs):
    """
    Print detection results
    """
    print(f"\nDetection: {class_name}")
    print(f"Confidence: {confidence*100:.1f}%")
    print("\nAll probabilities:")
    for i, cls in enumerate(CLASSES):
        prob = all_probs[i] * 100
        bar = "=" * int(prob / 2)
        print(f"  {cls:20s}: {prob:5.1f}% {bar}")

def camera_stream(camera_id=0):
    """
    Real-time defect detection from camera
    """
    cap = cv2.VideoCapture(camera_id)
    print(f"\nCamera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save temp frame
        cv2.imwrite('temp_frame.jpg', frame)

        # Detect
        start_time = time.time()
        class_name, confidence, _ = detect_defect('temp_frame.jpg')
        inference_time = (time.time() - start_time) * 1000

        # Color based on class
        if class_name == 'Clean':
            color = (0, 255, 0)  # Green
        elif class_name == 'Physical-Damage':
            color = (0, 0, 255)  # Red
        else:
            color = (0, 165, 255)  # Orange

        # Draw results
        text = f"{class_name}: {confidence*100:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, color, 2)
        cv2.putText(frame, f"{inference_time:.0f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('RGB Defect Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def batch_process(folder_path):
    """
    Process all images in a folder
    """
    import os
    from pathlib import Path

    folder = Path(folder_path)
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(folder.glob(ext))

    print(f"\nProcessing {len(images)} images...\n")

    results_summary = {cls: 0 for cls in CLASSES}

    for img_path in images:
        class_name, confidence, _ = detect_defect(str(img_path))
        results_summary[class_name] += 1
        print(f"{img_path.name:30s} -> {class_name:20s} ({confidence*100:.1f}%)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for cls, count in results_summary.items():
        print(f"{cls:20s}: {count}")

if __name__ == "__main__":
    import sys

    print("="*60)
    print("RGB Solar Panel Defect Detector")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Accuracy: 98.5%")
    print("="*60)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            # Single image
            class_name, confidence, all_probs = detect_defect(path)
            print_detection(class_name, confidence, all_probs)
        elif os.path.isdir(path):
            # Folder
            batch_process(path)
    else:
        # Demo
        print("\nUsage:")
        print("  Single image: python3 raspberry_pi_rgb.py image.jpg")
        print("  Batch:        python3 raspberry_pi_rgb.py folder/")
        print("  Camera:       Uncomment camera_stream() below")
        print("\n")

        # Uncomment for live camera
        # camera_stream()
