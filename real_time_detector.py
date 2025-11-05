import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import threading
from collections import deque

class RealTimeBeatDetector:
    def __init__(self, sample_rate=22050, block_size=1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.energy_history = deque(maxlen=100)
        self.beat_times = deque(maxlen=50)
        self.beat_energy = deque(maxlen=50)
        self.is_running = False
        self.beat_count = 0
        self.start_time = time.time()
        
        # Setup plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.setup_plots()
        
    def setup_plots(self):
        """Setup the real-time plots"""
        # Energy plot
        self.ax1.set_title('Real-time Energy Monitoring')
        self.ax1.set_ylabel('Energy')
        self.ax1.set_xlabel('Time (seconds)')
        self.ax1.grid(True, alpha=0.3)
        
        # Beat plot
        self.ax2.set_title('Beat Detection')
        self.ax2.set_ylabel('Beat Strength')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize lines
        self.energy_line, = self.ax1.plot([], [], 'b-', linewidth=1, label='Energy')
        self.threshold_line, = self.ax1.plot([], [], 'r--', linewidth=1, label='Threshold')
        self.beat_line, = self.ax2.plot([], [], 'ro-', markersize=8, label='Beats')
        
        self.ax1.legend()
        self.ax2.legend()
        
        plt.tight_layout()
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio input"""
        if status:
            print(f"Audio callback status: {status}")
        
        if self.is_running:
            audio = indata[:, 0]  # Use first channel
            energy = np.sum(audio ** 2)
            
            current_time = time.time() - self.start_time
            
            self.energy_history.append((current_time, energy))
            
            # Dynamic threshold
            if len(self.energy_history) > 10:
                energies = [e for t, e in self.energy_history]
                threshold = np.mean(energies) * 2.5
                
                # Detect beat
                if energy > threshold and len(self.energy_history) > 20:
                    self.beat_count += 1
                    self.beat_times.append(current_time)
                    self.beat_energy.append(energy)
                    print(f"BEAT #{self.beat_count} at {current_time:.2f}s - Energy: {energy:.4f}")
    
    def update_plot(self, frame):
        """Update the real-time plot"""
        if len(self.energy_history) > 0:
            times, energies = zip(*self.energy_history)
            
            # Update energy plot
            self.energy_line.set_data(times, energies)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Update threshold line
            if len(energies) > 10:
                threshold = np.mean(energies) * 2.5
                self.threshold_line.set_data(times, [threshold] * len(times))
            
            # Update beat plot
            if len(self.beat_times) > 0:
                self.beat_line.set_data(self.beat_times, self.beat_energy)
                self.ax2.relim()
                self.ax2.autoscale_view()
            
        return self.energy_line, self.threshold_line, self.beat_line
    
    def start_detection(self):
        """Start real-time beat detection"""
        print("Starting real-time beat detection...")
        print("Press 'q' to quit")
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size
        )
        
        self.stream.start()
        
        # Start animation
        self.ani = FuncAnimation(
            self.fig, self.update_plot, interval=50, blit=False
        )
        
        # Show plot (this will block until window is closed)
        plt.show()
        
    def stop_detection(self):
        """Stop real-time beat detection"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print(f"\nStopped. Total beats detected: {self.beat_count}")

def simple_real_time_detection(stop_flag=None):
    """Simplified real-time detection without plots"""
    print("Starting simple real-time beat detection...")
    print("Press 'Stop Real-time' in GUI to stop")
    
    energy_history = []
    beat_count = 0
    start_time = time.time()
    
    def audio_callback(indata, frames, time_info, status):
        nonlocal beat_count
        if status:
            print(status)
        
        # Check if we should stop
        if stop_flag and stop_flag():
            raise sd.CallbackStop()
        
        audio = indata[:, 0]
        energy = np.sum(audio ** 2)
        energy_history.append(energy)
        
        # Keep only recent history
        if len(energy_history) > 50:
            energy_history.pop(0)
        
        # Dynamic threshold
        if len(energy_history) > 10:
            threshold = np.mean(energy_history) * 2.0
            current_time = time.time() - start_time
            
            if energy > threshold and len(energy_history) > 20:
                beat_count += 1
                print(f"BEAT #{beat_count} at {current_time:.2f}s ♪")
    
    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=22050, blocksize=1024):
            while True:
                # Check stop flag periodically
                if stop_flag and stop_flag():
                    break
                time.sleep(0.1)
    except sd.CallbackStop:
        print("Real-time detection stopped via callback")
    except KeyboardInterrupt:
        print("\nReal-time detection interrupted")
    except Exception as e:
        print(f"Real-time detection error: {e}")
    finally:
        print(f"Stopped. Total beats detected: {beat_count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Beat Detection')
    parser.add_argument('--visual', action='store_true', help='Use visual real-time detection')
    parser.add_argument('--simple', action='store_true', help='Use simple text-based detection')
    
    args = parser.parse_args()
    
    if args.visual:
        detector = RealTimeBeatDetector()
        try:
            detector.start_detection()
        except KeyboardInterrupt:
            detector.stop_detection()
    else:
        simple_real_time_detection()

if __name__ == "__main__":
    main()