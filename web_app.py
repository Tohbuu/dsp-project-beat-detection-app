from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
from beat_detector import BeatDetector
import tempfile
import soundfile as sf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

detector = BeatDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    try:
        # Analyze the file
        results = detector.analyze_audio_file(filename, visualize=False)
        
        if results is None:
            return jsonify({'success': False, 'error': 'Analysis returned no results'}), 500
        
        # Extract tempo values safely
        tempo_energy = results.get('tempo_energy', 0)
        tempo_flux = results.get('tempo_flux', 0)
        
        # Calculate average, handle division by zero
        if tempo_energy > 0 and tempo_flux > 0:
            average_tempo = np.mean([tempo_energy, tempo_flux])
        elif tempo_energy > 0:
            average_tempo = tempo_energy
        else:
            average_tempo = tempo_flux if tempo_flux > 0 else 0
        
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)
        
        return jsonify({
            'success': True,
            'tempo_energy': round(float(tempo_energy), 1),
            'tempo_flux': round(float(tempo_flux), 1),
            'average_tempo': round(float(average_tempo), 1),
            'beat_count_energy': len(results.get('energy_beats', [])),
            'beat_count_flux': len(results.get('flux_beats', [])),
            'duration': round(float(results.get('audio_length', 0)), 2)
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filename):
            os.remove(filename)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/demo')
def create_demo():
    """Generate a demo beat file"""
    try:
        from demo_signal import create_demo_beat_signal
        
        tempo = request.args.get('tempo', 120, type=int)
        # Validate tempo
        if tempo < 40 or tempo > 240:
            tempo = 120
        
        filename = f"demo_{tempo}bpm.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        create_demo_beat_signal(filepath, tempo=tempo, duration=10)
        
        return send_file(filepath, as_attachment=True, download_name=filename, mimetype='audio/wav')
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)