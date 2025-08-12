#!/usr/bin/env python3
"""
Ventamin AI - Main Start Script
A comprehensive AI video generation system with analysis and video creation capabilities.
"""

import os
import sys
import subprocess
import importlib
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time

# Configure logging
# Create a custom StreamHandler that handles Unicode on Windows
class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Handle Unicode characters properly
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8')
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ventamin_ai.log', encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VentaminAIStarter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ventamin AI - Video Generator")
        self.root.geometry("600x500")
        self.root.configure(bg='#2c3e50')
        
        # Create directories
        self.setup_directories()
        
        # Setup UI
        self.setup_ui()
        
    def setup_directories(self):
        """Create necessary directories for the project"""
        directories = [
            'analysis_output',
            'generated_videos',
            'temp_frames',
            'logs',
            'config'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Ventamin AI Video Generator", 
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.status_label = ttk.Label(status_frame, text="Ready to start")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Start button
        self.start_button = ttk.Button(
            button_frame,
            text="Start Analysis & Video Generation",
            command=self.start_process,
            style='Accent.TButton'
        )
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        # Check dependencies button
        check_button = ttk.Button(
            button_frame,
            text="Check Dependencies",
            command=self.check_dependencies
        )
        check_button.grid(row=0, column=1, padx=(0, 10))
        
        # Open folders button
        folders_button = ttk.Button(
            button_frame,
            text="Open Output Folders",
            command=self.open_output_folders
        )
        folders_button.grid(row=0, column=2)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Process Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text widget for logs
        self.log_text = tk.Text(log_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def log_message(self, message):
        """Add message to log display"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        self.log_message("Checking dependencies...")
        
        required_packages = [
            'torch', 'transformers', 'opencv-python', 'moviepy',
            'numpy', 'Pillow', 'openai', 'pandas'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                self.log_message(f"[OK] {package} - Installed")
            except ImportError:
                missing_packages.append(package)
                self.log_message(f"[ERROR] {package} - Missing")
        
        if missing_packages:
            self.log_message(f"⚠️ Missing packages: {', '.join(missing_packages)}")
            self.log_message("💡 Click 'Install Dependencies' to install missing packages")
            
            # Show install button
            install_button = ttk.Button(
                self.root,
                text="📦 Install Missing Dependencies",
                command=lambda: self.install_dependencies(missing_packages)
            )
            install_button.grid(row=5, column=0, pady=(10, 0))
        else:
            self.log_message("✅ All dependencies are installed!")
    
    def install_dependencies(self, packages):
        """Install missing dependencies"""
        self.log_message("📦 Installing missing dependencies...")
        
        def install_thread():
            try:
                for package in packages:
                    self.log_message(f"Installing {package}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.log_message(f"✅ {package} installed successfully")
                
                self.log_message("🎉 All dependencies installed successfully!")
                messagebox.showinfo("Success", "All dependencies have been installed successfully!")
                
            except subprocess.CalledProcessError as e:
                self.log_message(f"❌ Error installing dependencies: {e}")
                messagebox.showerror("Error", f"Failed to install dependencies: {e}")
        
        threading.Thread(target=install_thread, daemon=True).start()
    
    def start_process(self):
        """Start the analysis and video generation process"""
        self.start_button.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Processing...")
        
        def process_thread():
            try:
                self.log_message("🚀 Starting Ventamin AI process...")
                
                # Step 1: Run analysis
                self.log_message("📊 Step 1: Running analysis...")
                self.run_analysis()
                
                # Step 2: Generate video
                self.log_message("🎬 Step 2: Generating video...")
                self.generate_video()
                
                # Step 2.5: Analyze Ventamin products
                self.log_message("📦 Step 2.5: Analyzing Ventamin products...")
                self.generate_tora_video()
                
                # Step 3: Create AI prompt
                self.log_message("🤖 Step 3: Creating AI prompt...")
                self.create_ai_prompt()
                
                self.log_message("🎉 Process completed successfully!")
                self.status_label.config(text="Completed successfully!")
                messagebox.showinfo("Success", "Analysis and video generation completed successfully!")
                
            except Exception as e:
                self.log_message(f"❌ Error during process: {e}")
                self.status_label.config(text="Error occurred")
                messagebox.showerror("Error", f"An error occurred: {e}")
            
            finally:
                self.progress.stop()
                self.start_button.config(state='normal')
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def run_analysis(self):
        """Run the analysis component"""
        try:
            # Import and run analysis
            from src.analysis.analyzer import VideoAnalyzer
            
            analyzer = VideoAnalyzer()
            analysis_result = analyzer.analyze_content()
            
            # Save analysis results
            analysis_file = Path("analysis_output/analysis_results.json")
            import json
            with open(analysis_file, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            self.log_message(f"✅ Analysis completed and saved to {analysis_file}")
            
        except Exception as e:
            self.log_message(f"❌ Analysis error: {e}")
            raise
    
    def generate_video(self):
        """Generate the video using Sora-style generation"""
        try:
            # Import and run Sora-style video generation
            from src.generators.sora_style_generator import SoraStyleGenerator
            
            generator = SoraStyleGenerator()
            generated_videos = generator.generate_ventamin_videos()
            
            if generated_videos:
                for video_path in generated_videos:
                    self.log_message(f"✅ Sora-style video generated: {video_path}")
            else:
                self.log_message("❌ No videos were generated")
            
        except Exception as e:
            self.log_message(f"❌ Video generation error: {e}")
            raise
    
    def generate_tora_video(self):
        """Generate video using Ventamin product analysis"""
        try:
            # Import and run Ventamin product analysis
            from src.analysis.ventamin_analyzer import VentaminAnalyzer
            
            analyzer = VentaminAnalyzer()
            analysis_results = analyzer.analyze_product_images()
            
            # Save analysis results
            analysis_file = Path("analysis_output/ventamin_analysis.json")
            import json
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            self.log_message(f"✅ Ventamin product analysis completed: {analysis_file}")
            
            # Generate summary
            summary = analyzer.generate_summary(analysis_results)
            self.log_message("📊 Product Analysis Summary:")
            for line in summary.split('\n'):
                if line.strip():
                    self.log_message(f"   {line}")
            
        except Exception as e:
            self.log_message(f"❌ Ventamin analysis error: {e}")
            raise
    
    def create_ai_prompt(self):
        """Create detailed AI prompt for video generation"""
        try:
            # Read analysis results
            analysis_file = Path("analysis_output/analysis_results.json")
            if analysis_file.exists():
                import json
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Create detailed prompt
                prompt_content = f"""
# Ventamin AI Video Generation Prompt

## Analysis Summary
{json.dumps(analysis_data, indent=2)}

## Video Generation Instructions
1. Use the analysis data above to create engaging video content
2. Focus on key insights and trends identified
3. Create visual elements that complement the analysis
4. Ensure smooth transitions and professional quality
5. Include appropriate background music and effects

## Technical Specifications
- Resolution: 1920x1080 (Full HD)
- Frame Rate: 30 fps
- Duration: 60-90 seconds
- Format: MP4 with H.264 encoding
- Audio: Stereo, 44.1kHz

## Style Guidelines
- Modern and professional appearance
- Consistent color scheme
- Clear typography and readability
- Smooth animations and transitions
- Engaging visual elements

## Content Structure
1. Introduction (10-15 seconds)
2. Main analysis presentation (40-60 seconds)
3. Key insights highlight (15-20 seconds)
4. Conclusion and call-to-action (10-15 seconds)
"""
                
                # Save prompt
                prompt_file = Path("analysis_output/ai_video_prompt.txt")
                with open(prompt_file, 'w') as f:
                    f.write(prompt_content)
                
                self.log_message(f"✅ AI prompt created: {prompt_file}")
            
        except Exception as e:
            self.log_message(f"❌ Error creating AI prompt: {e}")
            raise
    
    def open_output_folders(self):
        """Open output folders in file explorer"""
        folders = ['analysis_output', 'generated_videos']
        
        for folder in folders:
            folder_path = Path(folder)
            if folder_path.exists():
                try:
                    if sys.platform == "win32":
                        os.startfile(folder_path)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", folder_path])
                    else:
                        subprocess.run(["xdg-open", folder_path])
                except Exception as e:
                    self.log_message(f"❌ Error opening {folder}: {e}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = VentaminAIStarter()
    app.run() 