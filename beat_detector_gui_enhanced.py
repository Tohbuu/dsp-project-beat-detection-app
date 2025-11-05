# beat_detector_gui_enhanced.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import os
import time
from beat_detector import BeatDetector


class EnhancedBeatDetectorApp:
    def __init__(self, root):
        self.root = root
        self._setup_main_window()
        self._initialize_app_state()
        self._setup_styles()
        self._setup_gui()

    def _setup_main_window(self):
        """Configure the main application window (responsive, DPI-aware)"""
        # Basic window configuration
        self.root.title("🎵 Advanced DSP Beat Detection Analyzer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')

        # Make window responsive
        self.root.minsize(1000, 650)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Try to make fonts scale on high-DPI displays (best-effort)
        try:
            if hasattr(self.root, 'tk') and self.root.tk.call('tk', 'windowingsystem') == 'aqua':
                # macOS specifics handled by the platform
                pass
        except Exception:
            pass

    def _initialize_app_state(self):
        """Initialize application state variables"""
        self.detector = BeatDetector()
        self.current_file = None
        self.results = None
        self.current_figures = []
        self.realtime_running = False
        self.realtime_thread = None
        self.current_visualization_window = None
        self.visualization_history = []

        # Default subplot configuration
        self.subplot_config = {
            'rows': 5, 'cols': 1,
            'figsize_width': 15, 'figsize_height': 12,
            'show_audio_waveform': True, 'show_energy': True,
            'show_spectral_flux': True, 'show_tempo_over_time': True,
            'show_beat_intervals': True
        }

    def _setup_styles(self):
        """Configure modern styling for the application"""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            # fallback to default if clam isn't available
            style.theme_use(style.theme_names()[0])

        # Base widget colors
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#E0E0E0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=6)
        style.configure('TLabelframe', background='#2b2b2b', foreground='white')
        style.configure('TLabelframe.Label', background='#2b2b2b', foreground='white')
        style.configure('TNotebook', background='#2b2b2b')
        style.configure('TNotebook.Tab', background='#404040', foreground='white', padding=[8, 4])
        style.configure('Treeview', background='#1e1e1e', fieldbackground='#1e1e1e', foreground='white')
        style.map('TButton',
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', '!disabled', '#3a3a3a'), ('active', '#505050')])

    def _setup_gui(self):
        """Setup the enhanced GUI layout (responsive, resizable panes)"""
        # Root container (grid-based)
        root_container = ttk.Frame(self.root)
        root_container.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        root_container.columnconfigure(0, weight=1)
        root_container.rowconfigure(1, weight=1)

        # Title
        self._create_title_section(root_container)

        # Use a panedwindow for three main areas: left controls, center visual, right history
        paned = ttk.Panedwindow(root_container, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky='nsew', padx=2, pady=2)
        # left_frame: controls and file selection (collapsible feel)
        left_frame = ttk.Frame(paned, width=320)
        left_frame.columnconfigure(0, weight=1)
        # center_frame: visualization area (expands)
        center_frame = ttk.Frame(paned)
        center_frame.columnconfigure(0, weight=1)
        center_frame.rowconfigure(0, weight=1)
        # right_frame: history and quick actions
        right_frame = ttk.Frame(paned, width=300)
        right_frame.columnconfigure(0, weight=1)

        paned.add(left_frame, weight=0)
        paned.add(center_frame, weight=1)
        paned.add(right_frame, weight=0)

        # store references to frames for other methods to populate
        self._left_panel = left_frame
        self._center_panel = center_frame
        self._right_panel = right_frame

        # Build content in each pane using existing setup helpers (slim adaptions)
        # Left: analysis controls
        analysis_container = ttk.Frame(left_frame)
        analysis_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        # Recreate a compact file selection + options layout:
        self._create_file_selection_section(analysis_container)
        self._create_analysis_options_section(analysis_container)
        self._create_progress_section(analysis_container)

        # Add Real-time controls under analysis options
        realtime_frame = ttk.LabelFrame(analysis_container, text="🔴 Real-time Detection", padding="8")
        realtime_frame.pack(fill=tk.X, pady=(8, 4))
        ttk.Button(realtime_frame, text="🎤 Start Real-time", command=self.start_realtime).pack(side=tk.LEFT, padx=4)
        ttk.Button(realtime_frame, text="⏹ Stop Real-time", command=self.stop_realtime).pack(side=tk.LEFT, padx=4)
        ttk.Label(realtime_frame, text="Real-time mode listens to microphone or system audio.", font=('Arial', 8)).pack(fill=tk.X, pady=(6,0))

        # Center: visualization display
        viz_container = ttk.Frame(center_frame)
        viz_container.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        # Create a title bar for center area for quick actions
        viz_top_bar = ttk.Frame(viz_container)
        viz_top_bar.pack(fill=tk.X, pady=(0,6))
        ttk.Button(viz_top_bar, text="🔄 Generate Visualizations", command=self.generate_visualizations).pack(side=tk.LEFT, padx=4)
        ttk.Button(viz_top_bar, text="💾 Save Plots", command=self.save_plots).pack(side=tk.LEFT, padx=4)
        ttk.Button(viz_top_bar, text="🎛 Subplot Config", command=lambda: self.open_subplot_config(refresh=True)).pack(side=tk.LEFT, padx=4)
        ttk.Button(viz_top_bar, text="🔄 Refresh", command=self.refresh_current_visualization).pack(side=tk.LEFT, padx=4)
        # Visualization area (central placeholder)
        self.viz_canvas_container = ttk.Frame(viz_container)
        self.viz_canvas_container.pack(fill=tk.BOTH, expand=True)
        # Central placeholder: replaced when embedding figures
        self._clear_viz_canvas_placeholder()

        # Right: history & quick controls
        history_frame = ttk.LabelFrame(self._right_panel, text="🖼 Visualization History", padding=8)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.viz_history_list = tk.Listbox(history_frame, height=12, font=('Arial', 10),
                                          bg='#1e1e1e', fg='white', selectbackground='#4FC3F7')
        self.viz_history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,4))
        self.viz_history_list.bind('<<ListboxSelect>>', self.open_selected_visualization)
        viz_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.viz_history_list.yview)
        viz_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.viz_history_list.config(yscrollcommand=viz_scroll.set)

        # quick actions under history
        quick_frame = ttk.Frame(self._right_panel)
        quick_frame.pack(fill=tk.X, padx=6, pady=(0,6))
        ttk.Button(quick_frame, text="💾 Save Selected", command=lambda: self._save_history_selected()).pack(side=tk.LEFT, padx=3)
        ttk.Button(quick_frame, text="❌ Remove Selected", command=lambda: self._remove_history_selected()).pack(side=tk.LEFT, padx=3)
        ttk.Button(quick_frame, text="🗑 Clear History", command=lambda: self._clear_visualization_history()).pack(side=tk.LEFT, padx=3)

        # Bottom: results dock (docked but togglable size)
        results_pane = ttk.Frame(root_container)
        results_pane.grid(row=2, column=0, sticky='nsew', pady=(8,0))
        root_container.rowconfigure(2, weight=0)

        # A visible handle and compact label to remind user results are below
        results_handle = ttk.Frame(results_pane)
        results_handle.pack(fill=tk.X, padx=6)
        ttk.Label(results_handle, text="📈 Results — Use the toolbar buttons to copy/save", font=('Arial',9)).pack(side=tk.LEFT)

        # Place the actual results text in a collapsible frame
        self._results_frame = ttk.Frame(results_pane)
        self._results_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        # Create results text using your existing widget settings
        self.results_text = scrolledtext.ScrolledText(
            self._results_frame,
            wrap=tk.WORD,
            width=80,
            height=8,
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='#ffffff',
            insertbackground='white'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        # Results control row
        results_controls = ttk.Frame(self._results_frame)
        results_controls.pack(fill=tk.X, pady=(6,0))
        ttk.Button(results_controls, text="📋 Copy Results", command=self.copy_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(results_controls, text="💾 Save Results", command=self.save_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(results_controls, text="🧹 Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=4)

        # Keep a legacy notebook (tabs) available on demand; create it but keep it hidden
        self.notebook = ttk.Notebook(root_container)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.viz_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="🎼 Audio Analysis")
        self.notebook.add(self.viz_tab, text="📊 Visualization")
        self.notebook.add(self.results_tab, text="📈 Results")
        # Fill legacy tabs (so older code referencing them won't break)
        self._setup_visualization_tab()
        self._setup_results_tab()
        # By default, don't show the notebook. If you want it visible, you can call self.notebook.pack(...)
        # self.notebook.pack(fill=tk.BOTH, expand=True)

        # Small helper bind to resize placeholder when center pane resizes
        def _on_center_resize(event):
            # optional: adjust font/wraplength based on width
            pass
        self._center_panel.bind("<Configure>", _on_center_resize)

    def _create_title_section(self, parent):
        """Create the title section of the application (uses grid for responsiveness)"""
        title_frame = ttk.Frame(parent)
        title_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 8))
        title_frame.columnconfigure(0, weight=1)

        title_label = tk.Label(
            title_frame,
            text="🎵 ADVANCED BEAT DETECTION & TEMPO ANALYSIS",
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b', fg='#4FC3F7'
        )
        title_label.grid(row=0, column=0, sticky='w', padx=6)

        subtitle_label = tk.Label(
            title_frame,
            text="Digital Signal Processing Project - Multi-Algorithm Beat Detection",
            font=('Arial', 10),
            bg='#2b2b2b', fg='#B0BEC5'
        )
        subtitle_label.grid(row=1, column=0, sticky='w', padx=6)

    def _create_file_selection_section(self, parent):
        """Create file selection section"""
        file_frame = ttk.LabelFrame(parent, text="📁 Audio File Selection", padding="12")
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        file_grid = ttk.Frame(file_frame)
        file_grid.pack(fill=tk.X)
        file_grid.columnconfigure(0, weight=1)

        self.file_label = tk.Label(
            file_grid,
            text="No file selected",
            bg='#2b2b2b', fg='#E0E0E0', font=('Arial', 9),
            wraplength=600, justify=tk.LEFT, anchor='w'
        )
        self.file_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        btn_frame = ttk.Frame(file_grid)
        btn_frame.grid(row=0, column=1, sticky=tk.E)

        ttk.Button(btn_frame, text="Browse Audio File",
                  command=self.browse_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Create Demo Files",
                  command=self.create_demo_files).pack(side=tk.LEFT, padx=5)

    def _create_analysis_options_section(self, parent):
        """Create analysis options section with action buttons"""
        options_frame = ttk.LabelFrame(parent, text="⚙️ Analysis Options", padding="12")
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        analysis_btn_frame = ttk.Frame(options_frame)
        analysis_btn_frame.pack(fill=tk.X)

        btn_container = ttk.Frame(analysis_btn_frame)
        btn_container.pack(fill=tk.X, pady=5)

        # Analysis buttons
        analysis_buttons = [
            ("🚀 Run Basic Analysis", self.run_basic_analysis),
            ("🔬 Run Enhanced Analysis", self.run_enhanced_analysis)
        ]

        for text, command in analysis_buttons:
            ttk.Button(btn_container, text=text, command=command).pack(side=tk.LEFT, padx=(0, 10))

    def _create_progress_section(self, parent):
        """Create progress display section"""
        self.progress_frame = ttk.LabelFrame(parent, text="📊 Analysis Progress", padding="12")
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_label = tk.Label(
            self.progress_frame,
            text="Ready to analyze...",
            bg='#2b2b2b', fg='#E0E0E0', font=('Arial', 9),
            anchor='w'
        )
        self.progress_label.pack(fill=tk.X, anchor=tk.W)

        self.progress = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(5, 0))

    def _setup_visualization_tab(self):
        """Setup the visualization tab (legacy support — left mostly unused because we embed viz in center pane)"""
        # Keep compatibility: create a minimal frame to avoid breakage if any code references self.viz_tab
        viz_main_container = ttk.Frame(self.viz_tab)
        viz_main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        instruction = tk.Label(
            viz_main_container,
            text="Visualization pane is now embedded in the central workspace. Use the center pane to view and interact with plots.",
            bg='#2b2b2b', fg='#B0BEC5', font=('Arial', 11), justify=tk.CENTER
        )
        instruction.pack(fill=tk.BOTH, expand=True)

    def _setup_results_tab(self):
        """Setup the results tab (legacy, still available)"""
        # Keep a backup results tab for users who like tabs
        results_container = ttk.Frame(self.results_tab)
        results_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Recreate a read-only copy of results (sync with main results_text when updated)
        self.results_copy_text = scrolledtext.ScrolledText(
            results_container, wrap=tk.WORD, width=80, height=25,
            font=('Consolas', 10), bg='#1e1e1e', fg='#ffffff', insertbackground='white'
        )
        self.results_copy_text.pack(fill=tk.BOTH, expand=True)
        # Keep previous result buttons for compatibility
        results_controls = ttk.Frame(results_container)
        results_controls.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(results_controls, text="📋 Copy Results", command=self.copy_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(results_controls, text="💾 Save Results", command=self.save_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(results_controls, text="🧹 Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=(0, 10))

    def browse_file(self):
        """Browse for audio file"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a *.aac"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.current_file = filename
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            self.file_label.config(
                text=f"📄 {os.path.basename(filename)}\n"
                     f"📁 {os.path.dirname(filename)}\n"
                     f"💾 {file_size:.1f} MB"
            )
            self.update_progress("File selected - Ready for analysis")

    def create_demo_files(self):
        """Create demo beat files in a separate thread"""
        def create_demos():
            try:
                self.update_progress("Creating demo files...")
                from demo_signal import create_demo_beat_signal

                tempos = [90, 120, 140]
                for tempo in tempos:
                    self.update_progress(f"Creating {tempo} BPM demo file...")
                    create_demo_beat_signal(f"demo_{tempo}bpm.wav", tempo=tempo, duration=10)
                    time.sleep(0.5)

                self.update_progress("Demo files created successfully! ✅")
                messagebox.showinfo("Success", "Demo files created:\n• demo_90bpm.wav\n• demo_120bpm.wav\n• demo_140bpm.wav")

            except Exception as e:
                self.update_progress(f"Error creating demo files: {e}")
                messagebox.showerror("Error", f"Failed to create demo files: {e}")

        threading.Thread(target=create_demos, daemon=True).start()

    def run_basic_analysis(self):
        """Run basic beat detection analysis in a separate thread"""
        if not self._check_file_selected():
            return

        def analysis_thread():
            try:
                self.progress.start()
                self.update_progress("Starting basic analysis...")

                self.update_progress("Loading audio file...")
                self.results = self.detector.analyze_audio_file(self.current_file, visualize=False)

                if self.results:
                    self.root.after(0, self.display_basic_results)
                    self.update_progress("Basic analysis completed! ✅")
                else:
                    self.update_progress("Analysis failed - no results returned")

            except Exception as e:
                self._handle_analysis_error(e, "Basic analysis")
            finally:
                self.progress.stop()

        threading.Thread(target=analysis_thread, daemon=True).start()

    def run_enhanced_analysis(self):
        """Run enhanced beat detection analysis in a separate thread"""
        if not self._check_file_selected():
            return

        def analysis_thread():
            try:
                self.progress.start()
                self.update_progress("Starting enhanced analysis...")

                self.update_progress("Loading audio with enhanced features...")
                self.results = self.detector.analyze_audio_file_enhanced(self.current_file, visualize=False)

                if self.results:
                    self.root.after(0, self.display_enhanced_results)
                    self.update_progress("Enhanced analysis completed! ✅")
                else:
                    self.update_progress("Enhanced analysis failed - no results returned")

            except Exception as e:
                self._handle_analysis_error(e, "Enhanced analysis")
            finally:
                self.progress.stop()

        threading.Thread(target=analysis_thread, daemon=True).start()

    def _check_file_selected(self):
        """Check if a file is selected, show error if not"""
        if not self.current_file:
            messagebox.showerror("Error", "Please select an audio file first!")
            return False
        return True

    def _handle_analysis_error(self, error, analysis_type):
        """Handle analysis errors consistently"""
        self.update_progress(f"{analysis_type} error: {error}")
        self.root.after(0, lambda: messagebox.showerror("Error", f"{analysis_type} failed: {error}"))

    def display_basic_results(self):
        """Display basic analysis results in the results tab"""
        if not self.results:
            return

        results_text = self._format_basic_results()
        self._display_results_text(results_text)

    def display_enhanced_results(self):
        """Display enhanced analysis results in the results tab"""
        if not self.results:
            return

        results_text = self._format_enhanced_results()
        self._display_results_text(results_text)

    def _format_basic_results(self):
        """Format basic analysis results as text"""
        return f"""
🎵 BASIC BEAT DETECTION RESULTS
{'='*50}

📊 ANALYSIS SUMMARY:
• Audio Duration: {self.results['audio_length']:.2f} seconds
• Sample Rate: {self.detector.sample_rate} Hz

🎼 TEMPO ANALYSIS:
• Energy Method: {self.results['tempo_energy']:.1f} BPM
• Spectral Flux: {self.results['tempo_flux']:.1f} BPM
• Final Estimate: {np.mean([self.results['tempo_energy'], self.results['tempo_flux']]):.1f} BPM

🥁 BEAT DETECTION:
• Energy Beats: {len(self.results['energy_beats'])} beats
• Flux Beats: {len(self.results['flux_beats'])} beats
• Beat Density: {len(self.results['energy_beats'])/self.results['audio_length']:.2f} beats/sec

📈 BEAT INTERVALS:
• Min Interval: {np.min(np.diff(self.results['energy_beats'])):.3f}s
• Max Interval: {np.max(np.diff(self.results['energy_beats'])):.3f}s  
• Avg Interval: {np.mean(np.diff(self.results['energy_beats'])):.3f}s
• Consistency: {np.std(np.diff(self.results['energy_beats'])):.3f}s std dev

{'='*50}
        """

    def _format_enhanced_results(self):
        """Format enhanced analysis results as text"""
        # Calculate additional metrics
        tempo_stability = np.std(self.results['tempo_over_time']) if self.results.get('tempo_over_time') else 0
        tempo_range = f"{min(self.results['tempo_over_time']):.1f}-{max(self.results['tempo_over_time']):.1f}" if self.results.get('tempo_over_time') else "N/A"

        return f"""
🎵 ENHANCED BEAT DETECTION RESULTS
{'='*60}

📊 COMPREHENSIVE ANALYSIS:
• Audio Duration: {self.results['audio_length']:.2f} seconds
• Sample Rate: {self.detector.sample_rate} Hz
• Processing Method: Multi-Algorithm Fusion

🎼 ADVANCED TEMPO ANALYSIS:
• Primary Tempo: {self.results['final_tempo']:.1f} BPM
• Energy Method: {self.results['tempo_energy']:.1f} BPM  
• Spectral Flux: {self.results['tempo_flux']:.1f} BPM
• Tempo Range: {tempo_range} BPM
• Tempo Stability: {tempo_stability:.1f} BPM std dev

🥁 RHYTHMIC STRUCTURE:
• Total Beats: {len(self.results['energy_beats'])} beats
• Downbeats: {len(self.results.get('downbeats', []))} strong beats
• Weak Beats: {len(self.results.get('weak_beats', []))} weak beats
• Downbeat Ratio: {len(self.results.get('downbeats', []))/len(self.results['energy_beats'])*100:.1f}%

📈 BEAT PATTERN ANALYSIS:
• Beat Density: {len(self.results['energy_beats'])/self.results['audio_length']:.2f} beats/sec
• Dynamic Thresholding: ✅ Active
• Tempo Smoothing: ✅ Applied
• Downbeat Detection: ✅ Implemented

🎯 ALGORITHM PERFORMANCE:
• Energy Detection: {len(self.results['energy_beats'])} beats
• Spectral Flux: {len(self.results['flux_beats'])} beats  
• Algorithm Agreement: {'High' if abs(self.results['tempo_energy'] - self.results['tempo_flux']) < 20 else 'Moderate'}

{'='*60}
💡 MUSICAL INTERPRETATION:
This analysis suggests a {'consistent' if tempo_stability < 10 else 'varied'} rhythmic structure
with {'clear' if len(self.results.get('downbeats', [])) > len(self.results['energy_beats'])/8 else 'subtle'} downbeat emphasis.
        """

    def _display_results_text(self, results_text):
        """Display formatted results in the results text widget (and mirror to results tab)"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)
        # mirror to backup tab if exists
        try:
            if hasattr(self, 'results_copy_text'):
                self.results_copy_text.delete(1.0, tk.END)
                self.results_copy_text.insert(1.0, results_text)
        except Exception:
            pass
        # Also ensure notebook shows results tab if using the legacy notebook
        try:
            self.notebook.select(self.results_tab)
        except Exception:
            pass

    def _clear_viz_canvas_placeholder(self):
        """Clear the central viz container and show placeholder text"""
        for w in self.viz_canvas_container.winfo_children():
            w.destroy()
        self.viz_placeholder = tk.Label(
            self.viz_canvas_container,
            text="Run analysis then click 'Generate Visualizations'.\nThis central pane is resizable — drag the borders.",
            bg='#2b2b2b', fg='#757575', font=('Arial', 12),
            justify=tk.CENTER
        )
        self.viz_placeholder.pack(fill=tk.BOTH, expand=True)

    def generate_visualizations(self):
        """Generate visualizations with current configuration and embed into central pane"""
        if not self.results:
            messagebox.showwarning("Warning", "Please run analysis first!")
            return

        try:
            # Prepare audio data for visualization
            audio, sr = self.detector.load_audio(self.current_file)
            energy = self.detector.compute_energy(audio)
            spectral_flux = self.detector.compute_spectral_flux(audio)

            # Create figure with custom configuration
            fig = plt.figure(
                figsize=(self.subplot_config['figsize_width'],
                         self.subplot_config['figsize_height']),
                dpi=100  # Lower DPI for better performance
            )

            # Generate enhanced visualization (this returns a matplotlib Figure)
            fig = self.detector.visualize_enhanced_results(
                audio, sr, energy, spectral_flux,
                self.results['energy_beats'],
                self.results.get('downbeats', []),
                self.results.get('tempo_over_time', []),
                self.results.get('tempo_times', []),
                self.results.get('flux_beats', []),
                fig=fig
            )

            # Embed the figure in the central viz canvas container
            for w in self.viz_canvas_container.winfo_children():
                w.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_container)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

            toolbar_frame = ttk.Frame(self.viz_canvas_container)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            canvas.draw()

            # Store current figure for saving/export
            self.current_figures = [fig]

            # Add to history
            self._add_visualization_to_history(fig)

            # Add a small floating control to open in new window if desired
            open_btn = ttk.Button(self.viz_canvas_container, text="Open in Window", command=lambda: self._create_visualization_window(fig))
            open_btn.pack(side=tk.BOTTOM, anchor='ne', padx=6, pady=6)

            self.update_progress("Visualizations generated and embedded! ✅")
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {e}")

    def _add_visualization_to_history(self, fig):
        """Add current visualization to history"""
        viz_info = {
            'file': os.path.basename(self.current_file) if self.current_file else "Unknown",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'figure': fig,
            'config': self.subplot_config.copy()
        }
        self.visualization_history.append(viz_info)
        try:
            self.viz_history_list.insert(tk.END, f"{viz_info['file']} ({viz_info['timestamp']})")
        except Exception:
            pass

    def _create_visualization_window(self, fig):
        """Create a new window for visualization display (preserves original behavior)"""
        self.current_visualization_window = tk.Toplevel(self.root)
        self.current_visualization_window.title(f"Visualization - {os.path.basename(self.current_file) if self.current_file else 'untitled'}")
        self.current_visualization_window.geometry("1400x900")
        self.current_visualization_window.minsize(1000, 700)

        # Create main container with scrollbars
        main_container = ttk.Frame(self.current_visualization_window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create scrollable canvas
        viz_canvas = tk.Canvas(main_container, bg='#2b2b2b', highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=viz_canvas.yview)
        h_scrollbar = ttk.Scrollbar(main_container, orient="horizontal", command=viz_canvas.xview)

        scrollable_viz_frame = ttk.Frame(viz_canvas)
        scrollable_viz_frame.bind(
            "<Configure>",
            lambda e: viz_canvas.configure(scrollregion=viz_canvas.bbox("all"))
        )

        viz_canvas.create_window((0, 0), window=scrollable_viz_frame, anchor="nw")
        viz_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout for canvas and scrollbars
        viz_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        self._bind_mousewheel_scroll(viz_canvas)
        self._add_visualization_controls(scrollable_viz_frame, fig)
        self._embed_matplotlib_figure(scrollable_viz_frame, fig)

    def _add_visualization_controls(self, parent, fig):
        """Add control buttons to visualization window"""
        controls_frame = ttk.LabelFrame(parent, text="🎛️ Visualization Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        # First row of controls
        controls_row1 = ttk.Frame(controls_frame)
        controls_row1.pack(fill=tk.X, pady=5)

        # Control buttons
        control_buttons = [
            ("💾 Save Plot", lambda: self._save_current_plot(fig)),
            ("⚙️ Subplot Config", lambda: self.open_subplot_config(refresh=True)),
            ("🔄 Refresh", self.refresh_current_visualization),
            ("❌ Close Window", self.current_visualization_window.destroy)
        ]

        for i, (text, command) in enumerate(control_buttons):
            if i < 3:  # First three buttons on first row
                ttk.Button(controls_row1, text=text, command=command).pack(side=tk.LEFT, padx=5)

        # Second row with close button and info
        controls_row2 = ttk.Frame(controls_frame)
        controls_row2.pack(fill=tk.X, pady=5)

        ttk.Label(controls_row2, text="Use scrollbars or mouse wheel to navigate large plots").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_row2, text="❌ Close Window",
                  command=self.current_visualization_window.destroy).pack(side=tk.RIGHT, padx=5)

        # Configuration info
        config_info = ttk.Frame(controls_frame)
        config_info.pack(fill=tk.X, pady=5)

        config_text = f"Layout: {self.subplot_config['rows']}x{self.subplot_config['cols']} | Size: {self.subplot_config['figsize_width']}x{self.subplot_config['figsize_height']}in"
        ttk.Label(config_info, text=config_text, font=('Arial', 9)).pack(side=tk.LEFT)

    def _embed_matplotlib_figure(self, parent, fig):
        """Embed matplotlib figure in a given parent frame"""
        # Create canvas frame for matplotlib
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create matplotlib canvas
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Create navigation toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Draw the canvas
        canvas.draw()

    def _save_current_plot(self, fig):
        """Save the current plot to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {e}")

    def refresh_current_visualization(self):
        """Refresh the current visualization with updated settings"""
        # If an embedded canvas exists, destroy it and regenerate
        self._clear_viz_canvas_placeholder()
        self.generate_visualizations()

    def open_subplot_config(self, refresh=False):
        """Open subplot configuration dialog"""
        config_window = tk.Toplevel(self.root)
        config_window.title("⚙️ Subplot Configuration Tool")
        config_window.geometry("700x800")
        config_window.configure(bg='#2b2b2b')
        config_window.transient(self.root)
        config_window.grab_set()

        self._create_config_dialog_content(config_window, refresh)

    def _create_config_dialog_content(self, parent, refresh):
        """Create content for configuration dialog"""
        # Title
        title_label = tk.Label(parent,
                              text="Subplot Layout Configuration",
                              font=('Arial', 14, 'bold'),
                              bg='#2b2b2b', fg='#4FC3F7')
        title_label.pack(pady=10)

        # Configuration frame
        config_frame = ttk.LabelFrame(parent, text="Layout Settings", padding="15")
        config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create configuration widgets
        vars_dict = self._create_config_widgets(config_frame)

        # Preset layouts
        self._create_preset_layouts(config_frame, vars_dict)

        # Action buttons
        self._create_config_action_buttons(parent, vars_dict, refresh)

    def _create_config_widgets(self, parent):
        """Create configuration input widgets"""
        # Configuration variables
        vars_dict = {
            'rows': tk.IntVar(value=self.subplot_config['rows']),
            'cols': tk.IntVar(value=self.subplot_config['cols']),
            'width': tk.DoubleVar(value=self.subplot_config['figsize_width']),
            'height': tk.DoubleVar(value=self.subplot_config['figsize_height']),
            'audio': tk.BooleanVar(value=self.subplot_config['show_audio_waveform']),
            'energy': tk.BooleanVar(value=self.subplot_config['show_energy']),
            'flux': tk.BooleanVar(value=self.subplot_config['show_spectral_flux']),
            'tempo': tk.BooleanVar(value=self.subplot_config['show_tempo_over_time']),
            'intervals': tk.BooleanVar(value=self.subplot_config['show_beat_intervals'])
        }

        # Layout configuration
        layout_configs = [
            ("Number of Rows:", 'rows', 1, 10),
            ("Number of Columns:", 'cols', 1, 3),
            ("Figure Width (inches):", 'width', 5, 20),
            ("Figure Height (inches):", 'height', 5, 20)
        ]

        for text, var_key, from_val, to_val in layout_configs:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=5)
            tk.Label(frame, text=text, bg='#2b2b2b', fg='white').pack(side=tk.LEFT)
            spinbox = ttk.Spinbox(frame, from_=from_val, to=to_val, width=5,
                                 textvariable=vars_dict[var_key])
            spinbox.pack(side=tk.RIGHT)

        # Plot type selection
        plot_frame = ttk.LabelFrame(parent, text="Plot Types to Display", padding="10")
        plot_frame.pack(fill=tk.X, pady=10)

        plot_configs = [
            ("Audio Waveform", 'audio'),
            ("Energy Envelope", 'energy'),
            ("Spectral Flux", 'flux'),
            ("Tempo Over Time", 'tempo'),
            ("Beat Intervals", 'intervals')
        ]

        for text, var_key in plot_configs:
            cb = ttk.Checkbutton(plot_frame, text=text, variable=vars_dict[var_key])
            cb.pack(anchor=tk.W, pady=2)

        return vars_dict

    def _create_preset_layouts(self, parent, vars_dict):
        """Create preset layout buttons"""
        preset_frame = ttk.LabelFrame(parent, text="Quick Presets", padding="10")
        preset_frame.pack(fill=tk.X, pady=10)

        preset_configs = [
            ("Vertical Layout (5x1)", self._get_vertical_preset()),
            ("Grid Layout (3x2)", self._get_grid_preset()),
            ("Minimal Layout", self._get_minimal_preset())
        ]

        for text, preset in preset_configs:
            ttk.Button(preset_frame, text=text,
                      command=lambda p=preset: self._apply_preset(vars_dict, p)).pack(side=tk.LEFT, padx=5)

    def _get_vertical_preset(self):
        return {'rows': 5, 'cols': 1, 'width': 15, 'height': 12,
                'audio': True, 'energy': True, 'flux': True, 'tempo': True, 'intervals': True}

    def _get_grid_preset(self):
        return {'rows': 3, 'cols': 2, 'width': 16, 'height': 10,
                'audio': True, 'energy': True, 'flux': True, 'tempo': True, 'intervals': True}

    def _get_minimal_preset(self):
        return {'rows': 2, 'cols': 1, 'width': 12, 'height': 8,
                'audio': True, 'energy': True, 'flux': False, 'tempo': False, 'intervals': False}

    def _apply_preset(self, vars_dict, preset):
        """Apply preset configuration to variables"""
        for key, value in preset.items():
            if key in vars_dict:
                vars_dict[key].set(value)

    def _create_config_action_buttons(self, parent, vars_dict, refresh):
        """Create action buttons for configuration dialog"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        if refresh:
            ttk.Button(button_frame, text="Apply & Refresh",
                      command=lambda: self._apply_config_and_refresh(vars_dict, parent)).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Button(button_frame, text="Apply",
                      command=lambda: self._apply_config(vars_dict, parent)).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Reset Defaults",
                  command=lambda: self._reset_config_defaults(parent)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=parent.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply_config_and_refresh(self, vars_dict, parent):
        """Apply configuration and refresh visualization"""
        self._update_subplot_config(vars_dict)
        parent.destroy()
        self.refresh_current_visualization()

    def _apply_config(self, vars_dict, parent):
        """Apply configuration without refresh"""
        self._update_subplot_config(vars_dict)
        parent.destroy()
        messagebox.showinfo("Success", "Subplot configuration updated!\n\nApply this configuration by clicking 'Generate Visualizations'.")

    def _update_subplot_config(self, vars_dict):
        """Update subplot configuration from variables"""
        self.subplot_config.update({
            'rows': vars_dict['rows'].get(),
            'cols': vars_dict['cols'].get(),
            'figsize_width': vars_dict['width'].get(),
            'figsize_height': vars_dict['height'].get(),
            'show_audio_waveform': vars_dict['audio'].get(),
            'show_energy': vars_dict['energy'].get(),
            'show_spectral_flux': vars_dict['flux'].get(),
            'show_tempo_over_time': vars_dict['tempo'].get(),
            'show_beat_intervals': vars_dict['intervals'].get()
        })

    def _reset_config_defaults(self, parent):
        """Reset configuration to defaults"""
        self.subplot_config = {
            'rows': 5, 'cols': 1,
            'figsize_width': 15, 'figsize_height': 12,
            'show_audio_waveform': True, 'show_energy': True,
            'show_spectral_flux': True, 'show_tempo_over_time': True,
            'show_beat_intervals': True
        }
        parent.destroy()
        messagebox.showinfo("Success", "Configuration reset to defaults!")

    def open_selected_visualization(self, event):
        """Open the selected visualization from history"""
        selection = self.viz_history_list.curselection()
        if not selection:
            return

        idx = selection[0]
        viz_info = self.visualization_history[idx]
        fig = viz_info['figure']

        # Show the figure in a new window
        new_window = tk.Toplevel(self.root)
        new_window.title(f"Visualization - {viz_info['file']}")

        # Add controls
        controls_frame = ttk.Frame(new_window)
        controls_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(controls_frame, text="💾 Save Plot",
                  command=lambda: self._save_current_plot(fig)).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="❌ Close",
                  command=new_window.destroy).pack(side=tk.LEFT, padx=5)

        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, new_window)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.draw()

    def start_realtime(self):
        """Start real-time beat detection"""
        if self.realtime_running:
            messagebox.showinfo("Info", "Real-time detection is already running!")
            return

        def realtime_thread():
            try:
                self.realtime_running = True
                from real_time_detector import simple_real_time_detection

                # Create a proper stop flag that the real-time detector can check
                def should_stop():
                    return not self.realtime_running

                simple_real_time_detection(stop_flag=should_stop)

            except Exception as e:
                if self.realtime_running:  # Only show error if we didn't stop intentionally
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Real-time detection error: {e}"))
            finally:
                self.realtime_running = False
                self.root.after(0, lambda: self.update_progress("Real-time detection stopped"))

        # Start the thread
        self.realtime_thread = threading.Thread(target=realtime_thread, daemon=True)
        self.realtime_thread.start()
        self.update_progress("Real-time detection started... Speak or play music!")

    def stop_realtime(self):
        """Stop real-time beat detection"""
        if not self.realtime_running:
            messagebox.showinfo("Info", "Real-time detection is not running!")
            return

        self.realtime_running = False
        self.update_progress("Stopping real-time detection...")

    def update_progress(self, message):
        """Update progress label in a thread-safe manner"""
        def update():
            self.progress_label.config(text=message)
        self.root.after(0, update)

    def copy_results(self):
        """Copy results to clipboard"""
        results = self.results_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(results)
        self.update_progress("Results copied to clipboard! ✅")

    def save_results(self):
        """Save results to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.results_text.get(1.0, tk.END))
            self.update_progress(f"Results saved to {filename} ✅")

    def save_plots(self):
        """Save current plots to file"""
        if hasattr(self, 'current_figures') and self.current_figures:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                try:
                    self.current_figures[0].savefig(filename, dpi=300, bbox_inches='tight')
                    self.update_progress(f"Plot saved to {filename} ✅")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save plot: {e}")
        else:
            messagebox.showwarning("Warning", "No plots available to save")

    def clear_results(self):
        """Clear results from display"""
        self.results_text.delete(1.0, tk.END)
        try:
            if hasattr(self, 'results_copy_text'):
                self.results_copy_text.delete(1.0, tk.END)
        except Exception:
            pass
        self.update_progress("Results cleared")

    def _bind_mousewheel_scroll(self, canvas):
        """Bind mousewheel event for scrolling (supports Windows/macOS/Linux)"""
        def _on_mousewheel(event):
            # Windows / macOS produce event.delta with different signs/scales
            delta = 0
            if getattr(event, 'num', None) == 4:  # X11 scroll up
                delta = -1
            elif getattr(event, 'num', None) == 5:  # X11 scroll down
                delta = 1
            elif hasattr(event, 'delta'):
                # On Windows delta is multiples of 120; on macOS it can be small
                try:
                    delta = int(-1 * (event.delta / 120))
                except Exception:
                    delta = int(-1 * event.delta)
            try:
                canvas.yview_scroll(delta, "units")
            except Exception:
                try:
                    canvas.yview_scroll(int(delta), "units")
                except Exception:
                    pass

        # Bind both button-4/5 for X11 and <MouseWheel> for Windows/macOS
        try:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)
        except Exception:
            pass

    def _save_history_selected(self):
        sel = self.viz_history_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "No visualization selected to save.")
            return
        idx = sel[0]
        try:
            fig = self.visualization_history[idx]['figure']
            self._save_current_plot(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save selected visualization: {e}")

    def _remove_history_selected(self):
        sel = self.viz_history_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "No visualization selected.")
            return
        idx = sel[0]
        self.viz_history_list.delete(idx)
        del self.visualization_history[idx]
        self.update_progress("Removed selected visualization from history.")

    def _clear_visualization_history(self):
        if messagebox.askyesno("Confirm", "Clear the entire visualization history?"):
            self.viz_history_list.delete(0, tk.END)
            self.visualization_history.clear()
            self.update_progress("Visualization history cleared.")

    def on_closing(self):
        """Handle application closing"""
        # Stop real-time detection if running
        if self.realtime_running:
            self.realtime_running = False
            print("Stopping real-time detection...")

        # Close any open matplotlib figures
        plt.close('all')

        # Close visualization windows
        if hasattr(self, 'current_visualization_window') and self.current_visualization_window:
            try:
                self.current_visualization_window.destroy()
            except:
                pass

        # Destroy the root window
        self.root.destroy()

        # Force exit if needed
        import os
        os._exit(0)


def main():
    root = tk.Tk()
    app = EnhancedBeatDetectorApp(root)

    # Set up proper closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        app.on_closing()
    except Exception as e:
        print(f"Unexpected error: {e}")
        app.on_closing()


if __name__ == "__main__":
    main()
