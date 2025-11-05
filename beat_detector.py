import numpy as np
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
import soundfile as sf
import argparse
import os
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks


class BeatDetector:
    def __init__(self, sample_rate=22050, frame_size=1024, hop_size=512):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        
    def load_audio(self, file_path):
        """Load audio file and convert to mono"""
        print(f"Loading audio file: {file_path}")
        
        try:
            # Use librosa for all audio file types
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            print(f"Audio loaded: {len(audio)/sr:.2f} seconds, Sample rate: {sr} Hz")
            return audio, sr
            
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def bandpass_filter(self, audio, lowcut=100, highcut=4000):
        """Apply bandpass filter to focus on percussive elements"""
        print("Applying bandpass filter...")
        
        # Calculate Nyquist frequency
        nyquist = self.sample_rate / 2
        
        # Normalize frequencies to 0-1 range (as fraction of Nyquist)
        low_normalized = lowcut / nyquist
        high_normalized = highcut / nyquist
        
        # Ensure frequencies are within valid range (0 to 1)
        low_normalized = max(0.001, min(0.499, low_normalized))
        high_normalized = max(0.002, min(0.499, high_normalized))
        
        # Ensure low < high
        if low_normalized >= high_normalized:
            high_normalized = low_normalized + 0.01
        
        print(f"  Filter range: {lowcut}-{highcut} Hz")
        print(f"  Normalized: {low_normalized:.4f}-{high_normalized:.4f}")
        
        try:
            # Butterworth bandpass filter (lower order for stability)
            b, a = butter(2, [low_normalized, high_normalized], btype='band')
            filtered_audio = filtfilt(b, a, audio)
            print("  ✓ Filter applied successfully")
            return filtered_audio
        except Exception as e:
            print(f"  ⚠️  Filter failed: {e}")
            print("  Returning original audio")
            return audio
    
    def compute_energy(self, audio):
        """Compute energy envelope of the signal"""
        print("Computing energy envelope...")
        energy = []
        frames = len(audio) // self.hop_size
        
        for i in range(frames):
            start = i * self.hop_size
            end = start + self.frame_size
            if end < len(audio):
                frame = audio[start:end]
                energy.append(np.sum(frame ** 2))
        
        return np.array(energy)
    
    def compute_spectral_flux(self, audio):
        """Compute spectral flux for beat detection"""
        print("Computing spectral flux...")
        frames = len(audio) // self.hop_size
        flux = []
        prev_spectrum = None
        
        for i in range(frames):
            start = i * self.hop_size
            end = start + self.frame_size
            if end < len(audio):
                frame = audio[start:end]
                windowed = frame * np.hanning(len(frame))
                spectrum = np.abs(np.fft.fft(windowed)[:len(windowed)//2])
                
                if prev_spectrum is not None:
                    diff = spectrum - prev_spectrum
                    diff[diff < 0] = 0  # Only consider increases
                    flux.append(np.sum(diff))
                else:
                    flux.append(0)
                
                prev_spectrum = spectrum
        
        return np.array(flux)
    
    def detect_beats(self, energy_signal, threshold_factor=1.3, method='energy'):
        """Detect beats from energy signal with improved parameters"""
        # Use a combination of mean and median for robust thresholding
        mean_energy = np.mean(energy_signal)
        median_energy = np.median(energy_signal)
        threshold = max(mean_energy, median_energy) * threshold_factor

        # Adjust minimum distance based on expected tempo range
        # For music, reasonable range is 60-180 BPM
        max_bpm = 180  # Maximum expected BPM
        min_beat_distance = int((60 / max_bpm) * self.sample_rate / self.hop_size)

        # Find peaks with better parameters
        peaks, properties = find_peaks(
            energy_signal,
            height=threshold,
            distance=min_beat_distance,
            prominence=mean_energy*0.3,  # Ensure significant peaks
            width=2  # Minimum width of peaks
        )

        # Post-process: remove peaks that are too close together
        if len(peaks) > 1:
            final_peaks = [peaks[0]]
            for i in range(1, len(peaks)):
                if (peaks[i] - final_peaks[-1]) >= min_beat_distance:
                    final_peaks.append(peaks[i])
            peaks = np.array(final_peaks)

        print(f"Detected {len(peaks)} beats with {method} method")
        return peaks
    
    def estimate_tempo(self, beat_times, method='autocorrelation'):
        """Estimate tempo from beat intervals with octave error correction"""
        if len(beat_times) < 3:
            return 0
        
        # Calculate all beat intervals
        intervals = np.diff(beat_times)
        
        # Remove outliers (intervals that are too short or too long)
        median_interval = np.median(intervals)
        valid_intervals = intervals[(intervals > 0.3) & (intervals < 2.0)]
        
        if len(valid_intervals) == 0:
            return 0
        
        # Calculate BPM from median interval
        median_bpm = 60.0 / np.median(valid_intervals)
        
        # Common tempo ranges for music
        common_tempos = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        
        # Find the closest common tempo (correcting for octave errors)
        best_tempo = median_bpm
        best_error = abs(median_bpm - common_tempos[0])
        
        # Check normal tempo and possible octave errors (half and double)
        for multiplier in [0.5, 1.0, 2.0]:
            scaled_bpm = median_bpm * multiplier
            for common_tempo in common_tempos:
                error = abs(scaled_bpm - common_tempo)
                if error < best_error and 60 <= scaled_bpm <= 180:
                    best_error = error
                    best_tempo = common_tempo
        
        # If we have enough beats, use autocorrelation for more accuracy
        if method == 'autocorrelation' and len(beat_times) > 10:
            try:
                # Create a beat signal
                duration = beat_times[-1]
                time_resolution = 0.01  # 10ms resolution
                time_points = int(duration / time_resolution)
                beat_signal = np.zeros(time_points)
                
                for beat in beat_times:
                    idx = int(beat / time_resolution)
                    if idx < len(beat_signal):
                        beat_signal[idx] = 1
                
                # Compute autocorrelation
                correlation = np.correlate(beat_signal, beat_signal, mode='full')
                correlation = correlation[len(correlation)//2:]
                
                # Find peaks in reasonable tempo range (30-240 BPM)
                min_lag = int(60/240 / time_resolution)  # 240 BPM
                max_lag = int(60/30 / time_resolution)   # 30 BPM
                
                if max_lag < len(correlation):
                    correlation_region = correlation[min_lag:max_lag]
                    peaks, _ = find_peaks(correlation_region, 
                                        distance=int(60/240 / time_resolution),
                                        height=np.max(correlation_region)*0.3)
                    
                    if len(peaks) > 0:
                        main_peak = peaks[0] + min_lag
                        beat_period = main_peak * time_resolution
                        autocorr_bpm = 60.0 / beat_period
                        
                        # Choose the most reasonable tempo
                        if 60 <= autocorr_bpm <= 180:
                            best_tempo = autocorr_bpm
            except:
                pass  # Fall back to interval method if autocorrelation fails
        
        return best_tempo

    def estimate_tempo_improved(self, beat_times, method='autocorrelation'):
        """Improved tempo estimation that avoids subdivision errors"""
        if len(beat_times) < 3:
            return 0
        
        # Calculate all beat intervals
        intervals = np.diff(beat_times)
        
        # Remove outliers
        median_interval = np.median(intervals)
        valid_intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
        
        if len(valid_intervals) == 0:
            return 0
        
        # Calculate raw BPM from median interval
        raw_bpm = 60.0 / np.median(valid_intervals)
        
        # Common musical tempos (focus on typical pop/EDM ranges)
        common_tempos = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 
                         125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]
        
        # Check for tempo doubling/halving (common errors)
        candidate_tempos = []
        
        for multiplier in [0.5, 1.0, 2.0]:  # Check half, normal, and double tempo
            candidate = raw_bpm * multiplier
            
            # Only consider reasonable musical tempos
            if 60 <= candidate <= 180:
                # Find closest common tempo
                closest_tempo = min(common_tempos, key=lambda x: abs(x - candidate))
                error = abs(candidate - closest_tempo)
                candidate_tempos.append((closest_tempo, error, multiplier))
        
        if not candidate_tempos:
            return raw_bpm
        
        # Prefer the candidate with smallest error to common tempo
        candidate_tempos.sort(key=lambda x: x[1])  # Sort by error
        
        # But strongly prefer the 1.0 multiplier (original tempo) if error is reasonable
        normal_tempo_candidates = [c for c in candidate_tempos if c[2] == 1.0]
        if normal_tempo_candidates and normal_tempo_candidates[0][1] < 10:
            best_tempo = normal_tempo_candidates[0][0]
        else:
            best_tempo = candidate_tempos[0][0]
        
        # Additional check: if detected tempo is very high (>150), consider it might be doubled
        if best_tempo > 150:
            # Check if half tempo would be more musically common
            half_tempo = best_tempo / 2
            closest_half = min(common_tempos, key=lambda x: abs(x - half_tempo))
            if abs(half_tempo - closest_half) < 5:  # If half tempo is very close to a common tempo
                best_tempo = closest_half
        
        return best_tempo

    def estimate_tempo_advanced(self, beat_times, method='autocorrelation'):
        """Advanced tempo estimation with musical context awareness"""
        if len(beat_times) < 4:
            return 0
        
        # Calculate all intervals
        intervals = np.diff(beat_times)
        
        # Remove outliers using IQR method
        Q1 = np.percentile(intervals, 25)
        Q3 = np.percentile(intervals, 75)
        IQR = Q3 - Q1
        valid_intervals = intervals[(intervals >= Q1 - 1.5*IQR) & (intervals <= Q3 + 1.5*IQR)]
        
        if len(valid_intervals) < 3:
            return 0
        
        # Calculate BPM from median interval (most robust)
        median_interval = np.median(valid_intervals)
        raw_bpm = 60.0 / median_interval
        
        # Common music tempos across genres
        common_tempos = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 
                         125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190]
        
        # Check for common tempo errors (half, double, etc.)
        candidates = []
        
        for multiplier in [0.5, 0.667, 1.0, 1.5, 2.0]:
            candidate_bpm = raw_bpm * multiplier
            
            # Only consider musically reasonable tempos
            if 60 <= candidate_bpm <= 200:
                # Find closest common tempo
                closest_tempo = min(common_tempos, key=lambda x: abs(x - candidate_bpm))
                error = abs(candidate_bpm - closest_tempo)
                
                # Weight preference: strongly prefer 1.0 multiplier (original tempo)
                weight = 1.0 if multiplier == 1.0 else 0.7
                weighted_error = error * weight
                
                candidates.append((closest_tempo, weighted_error, multiplier, candidate_bpm))
        
        if not candidates:
            return raw_bpm
        
        # Sort by weighted error
        candidates.sort(key=lambda x: x[1])
        
        # Take the best candidate
        best_tempo, best_error, best_multiplier, best_raw = candidates[0]
        
        print(f"🎵 Advanced Tempo Analysis:")
        print(f"   Raw BPM: {raw_bpm:.1f}")
        print(f"   Best Candidate: {best_tempo} BPM")
        
        return best_tempo

    def analyze_audio_file(self, file_path, visualize=True):
        """Complete analysis of an audio file"""
        print(f"\n=== Analyzing: {file_path} ===")
        
        # Load and process audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
            
        # Update sample rate if different from loaded file
        if sr != self.sample_rate:
            self.sample_rate = sr
        
        audio = self.bandpass_filter(audio)
        
        # Compute features
        energy = self.compute_energy(audio)
        spectral_flux = self.compute_spectral_flux(audio)
        
        # Detect beats with appropriate thresholds
        energy_beats = self.detect_beats(energy, threshold_factor=1.2, method='energy')
        flux_beats = self.detect_beats(spectral_flux, threshold_factor=0.5, method='flux')  # Lower threshold for flux
        
        # Convert to time
        time_axis = np.arange(len(energy)) * self.hop_size / sr
        energy_beat_times = time_axis[energy_beats]
        flux_beat_times = time_axis[flux_beats]
        
        # Debug the intervals
        self.debug_beat_intervals(energy_beat_times, file_path)
        
        # Estimate tempo
        tempo_energy = self.estimate_tempo(energy_beat_times)
        tempo_flux = self.estimate_tempo(flux_beat_times) if len(flux_beat_times) > 1 else 0
        
        print(f"\n=== RESULTS ===")
        print(f"Tempo (Energy method): {tempo_energy:.1f} BPM")
        print(f"Tempo (Spectral Flux): {tempo_flux:.1f} BPM")
        print(f"Detected {len(energy_beat_times)} beats (Energy method)")
        print(f"Detected {len(flux_beat_times)} beats (Spectral Flux method)")
        
        if visualize:
            self.visualize_results(audio, sr, energy, spectral_flux, 
                                 energy_beats, flux_beats, time_axis,
                                 energy_beat_times, flux_beat_times)
        
        return {
            'tempo_energy': tempo_energy,
            'tempo_flux': tempo_flux,
            'energy_beats': energy_beat_times,
            'flux_beats': flux_beat_times,
            'audio_length': len(audio)/sr
        }
    
    def analyze_audio_file_enhanced(self, file_path, visualize=True):
        """Enhanced analysis with dynamic thresholding, tempo smoothing, and downbeat detection"""
        print(f"\n=== ENHANCED ANALYSIS: {file_path} ===")
        
        # Load and process audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
            
        if sr != self.sample_rate:
            self.sample_rate = sr
        
        audio = self.bandpass_filter(audio)
        
        # Compute features
        energy = self.compute_energy(audio)
        spectral_flux = self.compute_spectral_flux(audio)
        time_axis = np.arange(len(energy)) * self.hop_size / sr
        
        # Detect beats with dynamic thresholding
        energy_beats = self.detect_beats_dynamic(energy, 'energy')
        flux_beats = self.detect_beats_dynamic(spectral_flux, 'flux')
        
        energy_beat_times = time_axis[energy_beats]
        flux_beat_times = time_axis[flux_beats]
        
        # Downbeat detection
        downbeats, weak_beats = self.detect_downbeats(energy_beats, energy, time_axis)
        downbeat_times = time_axis[downbeats]
        
        # Tempo analysis over time
        energy_tempos, tempo_times = self.analyze_tempo_over_time(energy_beat_times)
        smoothed_tempos = self.smooth_tempo(energy_tempos) if energy_tempos else []
        
        # Final tempo estimates
        tempo_energy = self.estimate_tempo(energy_beat_times)
        tempo_flux = self.estimate_tempo(flux_beat_times) if len(flux_beat_times) > 1 else 0
        
        # Use energy tempo as primary, fallback to flux if needed
        final_tempo = tempo_energy if tempo_energy > 0 else tempo_flux
        
        print(f"\n=== ENHANCED RESULTS ===")
        print(f"Primary Tempo: {final_tempo:.1f} BPM")
        print(f"Energy Method: {tempo_energy:.1f} BPM")
        print(f"Flux Method: {tempo_flux:.1f} BPM")
        print(f"Downbeats Detected: {len(downbeats)}")
        print(f"Weak Beats: {len(weak_beats)}")
        
        if smoothed_tempos:
            print(f"Tempo Range: {min(smoothed_tempos):.1f}-{max(smoothed_tempos):.1f} BPM")
            print(f"Tempo Stability: {np.std(smoothed_tempos):.1f} BPM std dev")
        
        if visualize:
            self.visualize_enhanced_results(audio, sr, energy, spectral_flux,
                                          energy_beat_times, downbeat_times,
                                          smoothed_tempos, tempo_times,
                                          flux_beat_times)
        
        return {
            'final_tempo': final_tempo,
            'tempo_energy': tempo_energy,
            'tempo_flux': tempo_flux,
            'energy_beats': energy_beat_times,
            'flux_beats': flux_beat_times,
            'downbeats': downbeat_times,
            'weak_beats': weak_beats,
            'tempo_over_time': smoothed_tempos,
            'tempo_times': tempo_times,
            'audio_length': len(audio)/sr
        }
    
    def analyze_audio_file_enhanced_v2(self, file_path, visualize=True):
        """Version 2 with improved tempo estimation and downbeat detection"""
        print(f"\n=== ENHANCED ANALYSIS V2: {file_path} ===")
        
        # Load and process audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
            
        if sr != self.sample_rate:
            self.sample_rate = sr
        
        audio = self.bandpass_filter(audio)
        
        # Compute features
        energy = self.compute_energy(audio)
        spectral_flux = self.compute_spectral_flux(audio)
        time_axis = np.arange(len(energy)) * self.hop_size / sr
        
        # Detect beats with dynamic thresholding
        energy_beats = self.detect_beats_dynamic(energy, 'energy')
        flux_beats = self.detect_beats_dynamic(spectral_flux, 'flux')
        
        energy_beat_times = time_axis[energy_beats]
        flux_beat_times = time_axis[flux_beats]
        
        # Use improved tempo estimation
        tempo_energy = self.estimate_tempo_improved(energy_beat_times)
        tempo_flux = self.estimate_tempo_improved(flux_beat_times) if len(flux_beat_times) > 1 else 0
        
        # Use improved downbeat detection
        downbeats, weak_beats = self.detect_downbeats_improved(energy_beat_times, energy, time_axis, tempo_energy)
        downbeat_times = time_axis[downbeats] if hasattr(downbeats, '__len__') else np.array([])
        
        # Tempo analysis over time
        energy_tempos, tempo_times = self.analyze_tempo_over_time(energy_beat_times)
        smoothed_tempos = self.smooth_tempo(energy_tempos) if energy_tempos else []
        
        # Use energy tempo as primary, fallback to flux if needed
        final_tempo = tempo_energy if tempo_energy > 0 else tempo_flux
        
        print(f"\n=== ENHANCED RESULTS V2 ===")
        print(f"Primary Tempo: {final_tempo:.1f} BPM")
        print(f"Energy Method: {tempo_energy:.1f} BPM")
        print(f"Flux Method: {tempo_flux:.1f} BPM")
        print(f"Downbeats Detected: {len(downbeats)}")
        print(f"Weak Beats: {len(weak_beats)}")
        
        if smoothed_tempos:
            print(f"Tempo Range: {min(smoothed_tempos):.1f}-{max(smoothed_tempos):.1f} BPM")
            print(f"Tempo Stability: {np.std(smoothed_tempos):.1f} BPM std dev")
        
        if visualize:
            self.visualize_enhanced_results(audio, sr, energy, spectral_flux,
                                            energy_beat_times, downbeat_times,
                                            smoothed_tempos, tempo_times,
                                            flux_beat_times)
        
        return {
            'final_tempo': final_tempo,
            'tempo_energy': tempo_energy,
            'tempo_flux': tempo_flux,
            'energy_beats': energy_beat_times,
            'flux_beats': flux_beat_times,
            'downbeats': downbeat_times,
            'weak_beats': weak_beats,
            'tempo_over_time': smoothed_tempos,
            'tempo_times': tempo_times,
            'audio_length': len(audio)/sr
        }
    
    def analyze_audio_file_enhanced_v3(self, file_path, visualize=True):
        """Version 3 with improved algorithms for all music genres"""
        print(f"\n=== ENHANCED ANALYSIS V3: {os.path.basename(file_path)} ===")
        
        # Load and process audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
            
        if sr != self.sample_rate:
            self.sample_rate = sr
        
        audio = self.bandpass_filter(audio)
        
        # Compute features
        energy = self.compute_energy(audio)
        spectral_flux = self.compute_spectral_flux(audio)
        time_axis = np.arange(len(energy)) * self.hop_size / sr
        
        # Detect beats with dynamic thresholding
        energy_beats = self.detect_beats_dynamic(energy, 'energy')
        flux_beats = self.detect_beats_dynamic(spectral_flux, 'flux')
        
        energy_beat_times = time_axis[energy_beats]
        flux_beat_times = time_axis[flux_beats]
        
        # Downbeat detection
        downbeats, weak_beats = self.detect_downbeats_kpop_enhanced(energy_beats, energy, time_axis)
        downbeat_times = time_axis[downbeats]
        
        # Tempo analysis over time
        energy_tempos, tempo_times = self.analyze_tempo_over_time(energy_beat_times)
        smoothed_tempos = self.smooth_tempo(energy_tempos) if energy_tempos else []
        
        # Use advanced tempo estimation
        tempo_energy = self.estimate_tempo_advanced(energy_beat_times)
        tempo_flux = self.estimate_tempo_advanced(flux_beat_times) if len(flux_beat_times) > 1 else 0
        
        # Final tempo selection
        final_tempo = tempo_energy if tempo_energy > 0 else tempo_flux
        
        print(f"\n=== ENHANCED RESULTS V3 ===")
        print(f"Primary Tempo: {final_tempo:.1f} BPM")
        print(f"Energy Method: {tempo_energy:.1f} BPM")
        print(f"Flux Method: {tempo_flux:.1f} BPM")
        print(f"Downbeats Detected: {len(downbeats)}")
        print(f"Weak Beats: {len(weak_beats)}")
        
        if smoothed_tempos:
            print(f"Tempo Range: {min(smoothed_tempos):.1f}-{max(smoothed_tempos):.1f} BPM")
            print(f"Tempo Stability: {np.std(smoothed_tempos):.1f} BPM std dev")
        
        if visualize:
            self.visualize_enhanced_results(audio, sr, energy, spectral_flux,
                                            energy_beat_times, downbeat_times,
                                            smoothed_tempos, tempo_times,
                                            flux_beat_times)
        
        return {
            'final_tempo': final_tempo,
            'tempo_energy': tempo_energy,
            'tempo_flux': tempo_flux,
            'energy_beats': energy_beat_times,
            'flux_beats': flux_beat_times,
            'downbeats': downbeat_times,
            'weak_beats': weak_beats,
            'tempo_over_time': smoothed_tempos,
            'tempo_times': tempo_times,
            'audio_length': len(audio)/sr
        }
    
    def visualize_results(self, audio, sr, energy, spectral_flux, 
                         energy_beats, flux_beats, time_axis,
                         energy_beat_times, flux_beat_times):
        """Visualize the analysis results"""
        print("Generating visualization...")
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Original audio
        plt.subplot(4, 1, 1)
        time_audio = np.arange(len(audio)) / sr
        plt.plot(time_audio, audio, alpha=0.7)
        plt.title('Original Audio Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Energy with beats
        plt.subplot(4, 1, 2)
        plt.plot(time_axis, energy, label='Energy Envelope', linewidth=1)
        plt.plot(energy_beat_times, energy[energy_beats], 'ro', markersize=4, label='Detected Beats')
        plt.title('Energy-based Beat Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Spectral flux with beats
        plt.subplot(4, 1, 3)
        plt.plot(time_axis, spectral_flux, label='Spectral Flux', linewidth=1, color='orange')
        plt.plot(flux_beat_times, spectral_flux[flux_beats], 'ro', markersize=4, label='Detected Beats')
        plt.title('Spectral Flux-based Beat Detection')
        plt.xlabel('Time (s)')
        plt.ylabel('Spectral Flux')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Combined view
        plt.subplot(4, 1, 4)
        plt.plot(energy_beat_times, np.ones_like(energy_beat_times), 'ro', markersize=8, label='Energy Beats')
        plt.plot(flux_beat_times, 0.5 * np.ones_like(flux_beat_times), 'bo', markersize=8, label='Flux Beats')
        plt.yticks([0.5, 1.0], ['Flux', 'Energy'])
        plt.title('Beat Timeline Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def visualize_enhanced_results(self, audio, sr, energy, spectral_flux,
                                  energy_beat_times, downbeat_times,
                                  tempo_over_time, tempo_times, 
                                  flux_beat_times=None, fig=None):
        """Enhanced visualization with improved readability and layout"""
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.figure(figsize=(16, 14))
        axes = fig.subplots(5, 1, gridspec_kw={'hspace': 0.5})
        fig.suptitle("DSP Beat Detection & Tempo Analysis", fontsize=18, fontweight='bold', color='#1565C0')

        time_axis_audio = np.arange(len(audio)) / sr
        time_axis_features = np.arange(len(energy)) * self.hop_size / sr

        # Plot 1: Original audio with all beat types
        axes[0].plot(time_axis_audio, audio, alpha=0.8, linewidth=1.2, color='#37474F')
        axes[0].set_title('Audio Signal', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].set_ylabel('Amplitude', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Regular beats
        if len(energy_beat_times) > 0:
            axes[0].scatter(energy_beat_times, [0] * len(energy_beat_times), 
                            color='#FF5252', marker='o', s=30, alpha=0.7, label='Beats')
        # Downbeats
        if len(downbeat_times) > 0:
            axes[0].scatter(downbeat_times, [0] * len(downbeat_times), 
                            color='#B71C1C', marker='o', s=100, alpha=0.9, label='Downbeats')
        axes[0].legend(fontsize=11, loc='upper right')

        # Plot 2: Energy envelope with dynamic threshold
        axes[1].plot(time_axis_features, energy, color='#1976D2', linewidth=1.5, label='Energy')
        dynamic_thresh = self.dynamic_threshold(energy)
        axes[1].plot(time_axis_features, dynamic_thresh, 'r--', linewidth=1.2, label='Dynamic Threshold')
        if len(energy_beat_times) > 0:
            beat_indices = [np.argmin(np.abs(time_axis_features - t)) for t in energy_beat_times]
            beat_energies = [energy[i] for i in beat_indices]
            axes[1].scatter(energy_beat_times, beat_energies, color='#FF5252', s=40, alpha=0.8)
        axes[1].set_title('Energy Envelope & Dynamic Threshold', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_ylabel('Energy', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Spectral flux
        axes[2].plot(time_axis_features, spectral_flux, color='#FFA000', linewidth=1.5, label='Spectral Flux')
        if flux_beat_times is not None and len(flux_beat_times) > 0:
            flux_beat_indices = [np.argmin(np.abs(time_axis_features - t)) for t in flux_beat_times]
            flux_beat_values = [spectral_flux[i] for i in flux_beat_indices]
            axes[2].scatter(flux_beat_times, flux_beat_values, color='#7B1FA2', s=40, alpha=0.8, label='Flux Beats')
        axes[2].set_title('Spectral Flux', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('Spectral Flux', fontsize=12)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Tempo over time
        if tempo_over_time and len(tempo_over_time) > 0:
            axes[3].plot(tempo_times, tempo_over_time, color='#388E3C', linewidth=2, label='Tempo')
            axes[3].axhline(y=np.mean(tempo_over_time), color='#D32F2F', linestyle='--', 
                            label=f'Avg: {np.mean(tempo_over_time):.1f} BPM')
            axes[3].set_title('Tempo Analysis Over Time', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Time (s)', fontsize=12)
            axes[3].set_ylabel('Tempo (BPM)', fontsize=12)
            axes[3].set_ylim(max(60, np.min(tempo_over_time)-10), np.max(tempo_over_time)+10)
            axes[3].legend(fontsize=11)
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'Insufficient data for tempo analysis over time', 
                         ha='center', va='center', transform=axes[3].transAxes, fontsize=12)
            axes[3].set_title('Tempo Analysis Over Time', fontsize=14, fontweight='bold')
            axes[3].set_xlabel('Time (s)', fontsize=12)
            axes[3].set_ylabel('Tempo (BPM)', fontsize=12)

        # Plot 5: Beat intervals and downbeat pattern
        if len(energy_beat_times) > 1:
            intervals = np.diff(energy_beat_times)
            axes[4].plot(energy_beat_times[1:], intervals, 'bo-', markersize=5, linewidth=1.2, label='Beat Intervals')
            if len(downbeat_times) > 0:
                downbeat_intervals = []
                downbeat_times_plot = []
                for i, beat_time in enumerate(energy_beat_times[1:], 1):
                    if energy_beat_times[i-1] in downbeat_times:
                        downbeat_intervals.append(intervals[i-1])
                        downbeat_times_plot.append(beat_time)
                if downbeat_intervals:
                    axes[4].scatter(downbeat_times_plot, downbeat_intervals, 
                                    color='#B71C1C', s=80, label='Downbeat Intervals')
            axes[4].axhline(y=np.mean(intervals), color='#D32F2F', linestyle='--', 
                            label=f'Avg: {np.mean(intervals):.3f}s')
            axes[4].set_title('Beat Intervals (Downbeats Highlighted)', fontsize=14, fontweight='bold')
            axes[4].set_xlabel('Time (s)', fontsize=12)
            axes[4].set_ylabel('Interval (s)', fontsize=12)
            axes[4].legend(fontsize=11)
            axes[4].grid(True, alpha=0.3)
        else:
            axes[4].text(0.5, 0.5, 'Insufficient beats for interval analysis', 
                         ha='center', va='center', transform=axes[4].transAxes, fontsize=12)
            axes[4].set_title('Beat Intervals', fontsize=14, fontweight='bold')
            axes[4].set_xlabel('Time (s)', fontsize=12)
            axes[4].set_ylabel('Interval (s)', fontsize=12)

        # Improve tick label size for all axes
        for ax in axes:
            ax.tick_params(axis='both', labelsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        # Do NOT call plt.show() here!
        return fig

    def debug_beat_intervals(self, beat_times, filename):
        """Debug method to analyze beat intervals"""
        if len(beat_times) < 2:
            return
        
        intervals = np.diff(beat_times)
        
        # Determine expected tempo from filename
        if "90bpm" in filename:
            expected_tempo = 90
        elif "120bpm" in filename:
            expected_tempo = 120
        elif "140bpm" in filename:
            expected_tempo = 140
        else:
            expected_tempo = 120  # Default
        
        expected_interval = 60.0 / expected_tempo
        
        print(f"\n🔍 BEAT INTERVAL DEBUG for {filename}:")
        print(f"Expected interval: {expected_interval:.3f}s ({expected_tempo} BPM)")
        print(f"Detected {len(intervals)} intervals:")
        print(f"  Min: {np.min(intervals):.3f}s")
        print(f"  Max: {np.max(intervals):.3f}s") 
        print(f"  Mean: {np.mean(intervals):.3f}s")
        print(f"  Median: {np.median(intervals):.3f}s")
        
        detected_bpm = 60.0 / np.median(intervals)
        print(f"  Detected BPM: {detected_bpm:.1f}")

    def dynamic_threshold(self, signal, window_size=50):
        """Calculate dynamic threshold based on local signal characteristics"""
        threshold_signal = np.zeros_like(signal)
        
        for i in range(len(signal)):
            start = max(0, i - window_size // 2)
            end = min(len(signal), i + window_size // 2)
            
            window = signal[start:end]
            local_mean = np.mean(window)
            local_std = np.std(window)
            
            # Dynamic threshold: mean + scaled standard deviation
            threshold_signal[i] = local_mean + (local_std * 0.5)
        
        return threshold_signal

    def detect_beats_dynamic(self, energy_signal, method='energy'):
        """Detect beats with dynamic thresholding"""
        print(f"Using dynamic thresholding for {method} method...")
        
        # Calculate dynamic threshold
        dynamic_thresh = self.dynamic_threshold(energy_signal)
        
        # Minimum distance between beats (for 240 BPM max)
        min_beat_distance = int((60 / 240) * self.sample_rate / self.hop_size)
        
        # Find peaks that exceed dynamic threshold
        peaks, properties = find_peaks(
            energy_signal, 
            height=dynamic_thresh,
            distance=min_beat_distance,
            prominence=np.mean(energy_signal)*0.2
        )
        
        print(f"Detected {len(peaks)} beats with dynamic thresholding")
        return peaks
    
    def analyze_tempo_over_time(self, beat_times, window_size=8):
        """Analyze tempo changes over time using sliding windows"""
        if len(beat_times) < window_size + 1:
            return [self.estimate_tempo(beat_times)], beat_times[window_size//2:len(beat_times)-window_size//2]
        
        tempos = []
        window_centers = []
        
        for i in range(len(beat_times) - window_size):
            window_beats = beat_times[i:i + window_size]
            window_tempo = self.estimate_tempo(window_beats)
            
            # Only include reasonable tempo values
            if 60 <= window_tempo <= 200:
                tempos.append(window_tempo)
                window_centers.append(beat_times[i + window_size // 2])
        
        return tempos, window_centers

    def smooth_tempo(self, tempos, window_size=3):
        """Smooth tempo sequence using moving average"""
        if len(tempos) < window_size:
            return tempos
        
        smoothed = []
        for i in range(len(tempos)):
            start = max(0, i - window_size // 2)
            end = min(len(tempos), i + window_size // 2 + 1)
            window = tempos[start:end]
            smoothed.append(np.median(window))
        
        return smoothed
    
    def detect_downbeats(self, beat_times, energy_signal, time_axis):
        """Identify strong (downbeats) vs weak beats"""
        if len(beat_times) < 4:
            return np.array([]), np.array([])
        
        # Get energy values at beat positions
        beat_energies = energy_signal[beat_times]
        
        # Normalize energies
        beat_energies = (beat_energies - np.min(beat_energies)) / (np.max(beat_energies) - np.min(beat_energies))
        
        # Group beats into measures (assuming 4/4 time)
        downbeats = []
        weak_beats = []
        
        for i in range(len(beat_times)):
            if i % 4 == 0:  # Assume first beat of measure is strongest
                # Check if this beat has high energy relative to neighbors
                if i > 0 and i < len(beat_times) - 1:
                    local_avg = np.mean(beat_energies[max(0, i-2):min(len(beat_energies), i+3)])
                    if beat_energies[i] > local_avg * 1.2:
                        downbeats.append(beat_times[i])
                    else:
                        weak_beats.append(beat_times[i])
                else:
                    downbeats.append(beat_times[i])
            else:
                weak_beats.append(beat_times[i])
        
        print(f"Detected {len(downbeats)} downbeats and {len(weak_beats)} weak beats")
        return np.array(downbeats), np.array(weak_beats)
    
    def detect_downbeats_improved(self, beat_times, energy_signal, time_axis, expected_tempo=None):
        """Improved downbeat detection using energy patterns and musical knowledge"""
        if len(beat_times) < 8:  # Need enough beats for pattern recognition
            return np.array([]), beat_times
        
        # Get energy values at beat positions
        beat_indices = [np.argmin(np.abs(time_axis - t)) for t in beat_times]
        beat_energies = energy_signal[beat_indices]
        
        # Normalize energies
        if np.max(beat_energies) > np.min(beat_energies):
            beat_energies = (beat_energies - np.min(beat_energies)) / (np.max(beat_energies) - np.min(beat_energies))
        
        # Try to detect measure boundaries (4/4 time assumption)
        downbeats = []
        weak_beats = []
        
        # Use multiple strategies
        energy_threshold = np.mean(beat_energies) + 0.5 * np.std(beat_energies)
        
        for i in range(len(beat_times)):
            is_downbeat = False
            
            # Strategy 1: Every 4th beat (simple 4/4 assumption)
            if i % 4 == 0:
                is_downbeat = True
            
            # Strategy 2: High energy beats
            if beat_energies[i] > energy_threshold:
                is_downbeat = True
            
            # Strategy 3: Look for energy peaks in local context
            if i >= 2 and i < len(beat_times) - 2:
                local_energies = beat_energies[i-2:i+3]
                if beat_energies[i] == np.max(local_energies):
                    is_downbeat = True
            
            if is_downbeat:
                downbeats.append(beat_times[i])
            else:
                weak_beats.append(beat_times[i])
        
        # If we found too few downbeats, use simpler method
        if len(downbeats) < len(beat_times) / 8:
            downbeats = beat_times[::4]  # Every 4th beat
            weak_beats = [t for t in beat_times if t not in downbeats]
        
        print(f"Detected {len(downbeats)} downbeats and {len(weak_beats)} weak beats")
        return np.array(downbeats), np.array(weak_beats)
    
    def detect_downbeats_kpop_enhanced(self, beat_times, energy_signal, time_axis):
        """Enhanced downbeat detection for complex K-pop rhythms"""
        if len(beat_times) < 16:  # Need enough beats for pattern recognition
            return np.array([]), beat_times
        
        beat_indices = [np.argmin(np.abs(time_axis - t)) for t in beat_times]
        beat_energies = energy_signal[beat_indices]
        
        # Normalize energies
        if np.max(beat_energies) > np.min(beat_energies):
            beat_energies = (beat_energies - np.min(beat_energies)) / (np.max(beat_energies) - np.min(beat_energies))
        
        downbeats = []
        weak_beats = []
        
        # Multiple detection strategies for complex music
        for i in range(len(beat_times)):
            is_downbeat = False
            
            # Strategy 1: High energy peaks
            if beat_energies[i] > np.percentile(beat_energies, 75):
                is_downbeat = True
            
            # Strategy 2: Pattern-based (every 4th beat in 4/4 time)
            if i % 4 == 0 and i > 0:
                # Check if this aligns with energy peaks
                window_size = min(8, len(beat_times) - i)
                if window_size > 0:
                    local_peak = np.argmax(beat_energies[i:i+window_size]) == 0
                    if local_peak:
                        is_downbeat = True
            
            # Strategy 3: Context-aware (look for energy patterns)
            if i >= 4 and i < len(beat_times) - 4:
                # Check if this beat starts a new phrase
                prev_energy = np.mean(beat_energies[i-4:i])
                next_energy = np.mean(beat_energies[i:i+4])
                if beat_energies[i] > prev_energy * 1.3 and beat_energies[i] > next_energy * 0.8:
                    is_downbeat = True
            
            if is_downbeat:
                downbeats.append(beat_times[i])
            else:
                weak_beats.append(beat_times[i])
        
        # Post-processing: ensure reasonable downbeat count
        expected_downbeats = len(beat_times) // 4
        if len(downbeats) < expected_downbeats // 2:
            # Use simpler method as fallback
            downbeats = beat_times[::4]
            weak_beats = [t for t in beat_times if t not in downbeats]
        
        print(f"🎵 Enhanced Downbeat Detection:")
        print(f"   Total beats: {len(beat_times)}")
        print(f"   Downbeats: {len(downbeats)} ({len(downbeats)/len(beat_times)*100:.1f}%)")
        print(f"   Expected downbeats: ~{expected_downbeats}")
        
        return np.array(downbeats), np.array(weak_beats)
    
def real_time_beat_detection():
    """Real-time beat detection using microphone input"""
    print("Starting real-time beat detection...")
    print("Press Ctrl+C to stop")
    
    detector = BeatDetector()
    energy_history = []
    beat_count = 0
    
    def audio_callback(indata, frames, time, status):
        nonlocal beat_count
        if status:
            print(status)
        
        audio = indata[:, 0]  # Use first channel
        energy = np.sum(audio ** 2)
        energy_history.append(energy)
        
        # Keep only recent history
        if len(energy_history) > 50:
            energy_history.pop(0)
        
        # Dynamic threshold based on recent history
        if len(energy_history) > 10:
            threshold = np.mean(energy_history) * 2.0
            if energy > threshold and len(energy_history) > 20:
                beat_count += 1
                print(f"BEAT #{beat_count}! ♪ Energy: {energy:.4f}")
    
    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=22050, blocksize=1024):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print(f"\nStopped. Total beats detected: {beat_count}")

def main():
    parser = argparse.ArgumentParser(description='Beat Detection and Tempo Estimation')
    parser.add_argument('--file', type=str, help='Audio file to analyze')
    parser.add_argument('--realtime', action='store_true', help='Run real-time beat detection')
    
    args = parser.parse_args()
    
    detector = BeatDetector()
    
    if args.realtime:
        real_time_beat_detection()
    elif args.file:
        if os.path.exists(args.file):
            results = detector.analyze_audio_file(args.file)
            if results:
                if results['tempo_flux'] > 0:
                    final_tempo = np.mean([results['tempo_energy'], results['tempo_flux']])
                else:
                    final_tempo = results['tempo_energy']  # Use energy method if flux fails

                print(f"\nFinal Tempo Estimate: {final_tempo:.1f} BPM")
        else:
            print(f"File not found: {args.file}")
    else:
        print("Beat Detection System Ready!")
        print("\nUsage options:")
        print("1. Analyze audio file: python beat_detector.py --file <audio_file>")
        print("2. Real-time detection: python beat_detector.py --realtime")
        print("3. Demo with test signal: python beat_detector.py --file demo")

if __name__ == "__main__":
    main()