# Virtual Environment Setup

This project has been set up with a Python virtual environment containing all the required dependencies.

## Quick Start

### Option 1: Use the activation script
```bash
./activate_env.sh
```

### Option 2: Manual activation
```bash
source venv/bin/activate
```

## Deactivating the Environment
```bash
deactivate
```

## Installed Packages

The following major packages have been installed:

- **FastAPI** (0.109.2) - Web framework
- **Uvicorn** (0.27.1) - ASGI server
- **Pydantic** (2.6.1) - Data validation
- **Supabase** - Database client
- **PyTorch** (2.8.0) - Deep learning framework
- **TensorFlow** (2.20.0) - Machine learning framework
- **Transformers** (4.51.3) - Hugging Face transformers
- **NeMo Toolkit** (2.4.0) - NVIDIA NeMo for TTS
- **Google Generative AI** (0.8.5) - Google's AI models
- **SoundFile** (0.13.1) - Audio file handling
- **MoviePy** (2.2.1) - Video editing
- **Pandas** (2.3.2) - Data manipulation
- **NumPy** (1.26.4) - Numerical computing

## Notes

- The virtual environment is located in the `venv/` directory
- All packages were installed from `requirements.txt`
- Some packages may have compatibility issues on Apple Silicon Macs
- If you encounter issues with TensorFlow, consider using TensorFlow for macOS or TensorFlow Metal

## Running the Application

Once the virtual environment is activated, you can run your application:

```bash
python main.py
# or
python app.py
```

## Troubleshooting

If you encounter issues:

1. Make sure the virtual environment is activated (you should see `(venv)` in your terminal prompt)
2. Try reinstalling specific packages: `pip install --force-reinstall package_name`
3. For TensorFlow issues on macOS, consider using: `pip install tensorflow-macos`
