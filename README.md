# Ventamin AI - Intelligent Video Generation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-green.svg)](https://opencv.org)
[![MoviePy](https://img.shields.io/badge/MoviePy-2.1+-orange.svg)](https://zulko.github.io/moviepy/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎬 What is Ventamin AI?

Ventamin AI is an intelligent video generation system that analyzes content and creates professional videos automatically. It combines content analysis, AI-powered insights, and advanced video generation to help you create engaging videos without needing deep technical knowledge.

**Key Features:**
- 📊 **Content Analysis**: Automatically analyzes video content for structure, visual elements, and engagement patterns
- 🎬 **AI Video Generation**: Creates professional videos using Sora-style generation techniques
- 🤖 **Smart Recommendations**: Provides AI-powered insights for better video creation
- 🎨 **Professional Output**: Generates high-quality MP4 videos with modern design principles
- 📱 **Easy-to-Use Interface**: Simple GUI for non-technical users

## 🚀 Quick Start

### Prerequisites

- **Windows 10/11** (or Mac/Linux)
- **Python 3.8 or newer**
- **At least 4GB free disk space**

### Installation

1. **Clone or download** this repository
2. **Navigate to the project directory**:
   ```bash
   cd "path/to/An-automatic-market-research-AI-with-AI-ads-generator-main"
   ```
3. **Install dependencies**:
   ```bash
   py -m pip install opencv-python moviepy pillow numpy pandas scikit-learn matplotlib seaborn
   ```
4. **Run the system**:
   ```bash
   py start_ventamin_ai.py
   ```

### First Run Setup

1. **Launch the application**: Double-click `start_ventamin_ai.py` or run from command line
2. **Check dependencies**: Click "Check Dependencies" to verify all packages are installed
3. **Create sample assets**: Run `py create_sample_images.py` to generate product images
4. **Start generation**: Click "Start Analysis & Video Generation"

## 🎯 How It Works

### 1. Content Analysis Phase
The system analyzes your content to understand:
- **Visual Elements**: Colors, fonts, layout, and design patterns
- **Content Structure**: Scene organization, pacing, and flow
- **Audio Characteristics**: Music, voice-over, and sound quality
- **Engagement Patterns**: What keeps viewers interested

### 2. AI Video Generation Phase
Using the analysis results, the system creates:
- **Professional Videos**: Modern design with smooth transitions
- **Sora-Style Content**: AI-powered video generation
- **Brand-Consistent Output**: Maintains visual identity throughout

### 3. AI Recommendations
The system provides:
- **Style Guidelines**: Color schemes, typography, and pacing recommendations
- **Content Suggestions**: Ideas for future video improvements
- **Technical Insights**: Quality metrics and optimization tips

## 📁 Project Structure

```
Ventamin AI/
├── 📁 assets/                     # Product images and assets
│   └── 📁 ventamin_assets/       # Sample product images
├── 📁 analysis_output/            # Analysis results and reports
├── 📁 generated_videos/           # Created videos
├── 📁 src/                        # Source code
│   ├── 📁 analysis/               # Content analysis tools
│   ├── 📁 generators/             # Video generation engines
│   ├── 📁 core/                   # Core system components
│   ├── 📁 utils/                  # Utility functions
│   └── 📁 config/                 # Configuration files
├── 📁 logs/                       # System logs
├── 📁 temp_frames/                # Temporary processing files
├── 🚀 start_ventamin_ai.py        # Main application (START HERE!)
├── 📦 create_sample_images.py     # Asset creation script
├── 🧪 test_system.py              # System verification script
└── 📋 README.md                   # This file
```

## 🎮 Usage Guide

### Basic Usage

1. **Launch the Application**
   ```bash
   py start_ventamin_ai.py
   ```

2. **Check System Status**
   - Click "Check Dependencies" to verify all packages are installed
   - Review the system status in the log window

3. **Generate Videos**
   - Click "Start Analysis & Video Generation"
   - Watch the progress in the log window
   - Check results in the `generated_videos` folder

4. **View Results**
   - **Videos**: Located in `generated_videos/`
   - **Analysis**: Check `analysis_output/` for detailed reports
   - **Logs**: Review `logs/` for system information

### Advanced Usage

#### Custom Asset Creation
```bash
# Create custom product images
py create_sample_images.py

# Or modify the script to create your own assets
```

#### System Testing
```bash
# Verify all components are working
py test_system.py

# Run specific tests
py test_final_system.py
py test_sora_style_video.py
```

#### Configuration
- Modify `src/config/settings.py` for custom settings
- Adjust video parameters in generator files
- Customize analysis parameters in analyzer modules

## 🔧 Troubleshooting

### Common Issues

#### Issue: "No module named 'cv2'"
**Solution**: Install OpenCV
```bash
py -m pip install opencv-python
```

#### Issue: "MoviePy import error"
**Solution**: Install MoviePy
```bash
py -m pip install moviepy
```

#### Issue: "Unicode encoding errors"
**Solution**: The system now handles Unicode properly. If issues persist, check your terminal encoding settings.

#### Issue: "Missing assets"
**Solution**: Run the asset creation script
```bash
py create_sample_images.py
```

#### Issue: "Video generation failed"
**Solutions**:
1. Ensure you have at least 2GB free disk space
2. Check that antivirus isn't blocking the application
3. Try running as administrator
4. Verify all dependencies are installed

### Getting Help

1. **Check the logs**: Look in the `logs/` folder for detailed error messages
2. **Run system test**: Use `py test_system.py` to diagnose issues
3. **Review console output**: The application shows detailed information during execution
4. **Check file permissions**: Ensure the application can write to output directories

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.8 or newer
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Graphics**: Any modern GPU (optional, for faster processing)

### Recommended Requirements
- **OS**: Windows 11 or macOS 12+
- **Python**: 3.10 or newer
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **Graphics**: NVIDIA GPU with 4GB+ VRAM (for AI features)

## 🎨 Customization

### Adding Your Own Content

1. **Product Images**: Place your images in `assets/ventamin_assets/`
2. **Video Templates**: Modify generator files in `src/generators/`
3. **Analysis Rules**: Customize analyzers in `src/analysis/`
4. **Styling**: Adjust colors and fonts in configuration files

### Creating Custom Generators

1. **Extend Base Classes**: Inherit from existing generator classes
2. **Implement Required Methods**: Follow the established interface
3. **Add to Main System**: Register your generator in the main application
4. **Test Thoroughly**: Use the test scripts to verify functionality

## 🔒 Security & Privacy

- **Local Processing**: All video generation happens on your local machine
- **No Data Upload**: Your content never leaves your computer
- **Open Source**: Review the code to ensure it meets your security requirements
- **Dependencies**: Only uses well-established, trusted Python packages

## 📈 Performance Tips

1. **Close Other Applications**: Free up RAM for video processing
2. **Use SSD Storage**: Faster read/write speeds for video files
3. **Optimize Images**: Use appropriately sized images (400x600px recommended)
4. **Batch Processing**: Generate multiple videos in one session
5. **Monitor Resources**: Check system performance during generation

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Use the issue tracker for bugs and feature requests
2. **Submit Pull Requests**: Share your improvements and fixes
3. **Improve Documentation**: Help make the system easier to use
4. **Test on Different Platforms**: Ensure cross-platform compatibility

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Run tests**: `py test_system.py`
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV Community**: For computer vision capabilities
- **MoviePy Team**: For video processing tools
- **Python Community**: For the amazing ecosystem of packages
- **AI Research Community**: For inspiration in video generation techniques

## 📞 Support

If you need help:

1. **Check this README** first - it covers most common issues
2. **Review the logs** for detailed error information
3. **Run the test scripts** to diagnose problems
4. **Check the troubleshooting section** above

## 🎉 Success Stories

Users have successfully created:
- **Product Demonstrations**: Showcase products with professional quality
- **Marketing Videos**: Engaging content for social media and websites
- **Training Materials**: Educational content with consistent branding
- **Brand Presentations**: Professional company overviews

---

**Happy Video Creating! 🎬✨**

*Built with ❤️ using Python, OpenCV, and MoviePy*
