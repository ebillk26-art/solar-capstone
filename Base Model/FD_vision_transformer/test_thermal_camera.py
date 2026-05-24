
import board, busio, adafruit_mlx90640
import numpy as np, cv2, time

print("Testing MLX90640 thermal camera...")
try:
    i2c    = busio.I2C(board.SCL, board.SDA, frequency=400000)
    camera = adafruit_mlx90640.MLX90640(i2c)
    camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
    buf    = [0] * 768
    print("Connected! Reading 5 frames...")
    for i in range(5):
        camera.getFrame(buf)
        temps = np.array(buf).reshape(24,32)
        print(f"  Frame {i+1}: min={temps.min():.1f}C  max={temps.max():.1f}C  mean={temps.mean():.1f}C")
        time.sleep(0.6)
    print("Camera is working correctly!")
except Exception as e:
    print(f"FAILED: {e}")
    print("Check wiring: VCC->Pin1(3.3V) GND->Pin6 SDA->Pin3 SCL->Pin5")
