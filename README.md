# Marine_Biology_GenAI


A project dedicated to leveraging generative AI for the creation and enhancement of marine biology resources. This repository contains tools to generate 2D images and 3D models of marine creatures, aiding in research, education, and conservation efforts.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Contributing](#contributing)
- [Future Goals](#future-goals)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project focuses on developing AI-powered 3D models of marine species for educational and research applications. The system converts user text prompts into realistic 2D images using either Stable Diffusion 3.5 or real-world images from FathomNet for underrepresented species. These images are then transformed into interactive 3D assets using Tencent Hunyuan3D models, producing `.glb` files that can be rendered in Blender, Unity, or AR applications. The solution includes a web-based visualizer and a Unity-based interface to display both textured and mesh views, all deployed on RunPod for scalable access. The goal is to create dynamic, high-quality 3D representations of marine life for immersive learning experiences and scientific exploration.

---

## Key Features
- **Dual-Mode Image Generation:** Translates user input into 2D images with two methods—Stable Diffusion 3.5 for generative images and FathomNet for actual images when unusual or underrepresented species are asked for.
- **Text-to-3D Conversion Pipeline:** Renders 3D GLB models from 2D images with Tencent Hunyuan3D models, delivering assets that are ready to use within Blender, Unity, or AR/VR projects.
-  **End-to-End Deployment:** Complete containerization solution deployed on RunPod with API endpoints to allow easy model execution on the cloud.
- **Interactive Web Visualizer:** Client based on HTML/CSS/JS for rendering 2D images and 3D models within the browser itself with support for viewing textured or mesh-only models.
- **Unity Integration:** Unity-based visualizer for interactive exploration and AR/VR-enabled future educational applications.
- **Data Extensibility:** Framework prepared for eventual dataset generation—users can retrieve stored 3D assets from the server or create new ones dynamically.
- **Education-Oriented Design:** Models and tools designed for incorporation within high school biology syllabuses, rendering marine biology study interactive and fun.

---

## File Structure

```
Marine_Biology_GenAI/
├── hy3dgen/                # Scripts for 3D model generation
├── image_model/            # Scripts and models for 2D image generation
├── txt2_3D/                # Scripts for text-to-3D conversion
├── output/                 # Default directory for generated outputs
├── .gitignore
├── README.md
├── requirements.txt
└── zip_conversion.py
```

---

## Getting Started

### Prerequisites

- Python ≥3.8
- CUDA-enabled GPU (recommended 24GB VRAM for Hunyuan3D 2)
- Git
- pip

---

### Installation

#### Install Python Packages

```bash
git clone https://github.com/Dhruvanshu-Joshi/Marine_Biology_GenAI.git
cd Marine_Biology_GenAI
pip install -r requirements.txt
```
---

#### Build Custom Rasterizer

```bash
cd hy3dgen/texgen/custom_rasterizer
python setup.py bdist_wheel
pip install dist/custom_rasterizer*.whl
```

Verify installation:

```bash
find  -name "*custom_rasterizer_kernel*.so"
```

Add to Python path:

```python
import sys
sys.path.append('path to custom_rasterizer_kernel.cpython-311-x86_64-linux-gnu.so (output returned by the find)')
```

---

#### Localize Pretrained Models

**Hunyuan3D 2 models:**

```python
from huggingface_hub import snapshot_download

# Download mesh and paint pipelines
snapshot_download(
    repo_id="tencent/Hunyuan3D-2",
    local_dir="models",
    allow_patterns=["hunyuan3d-dit-v2-0/model.fp16.safetensors",
                    "hunyuan3d-dit-v2-0/config.yaml"]
)

snapshot_download(
    repo_id="tencent/Hunyuan3D-2",
    local_dir="models",
    allow_patterns=["hunyuan3d-paint-v2-0-turbo/**"]
)

snapshot_download(
    repo_id="tencent/Hunyuan3D-2",
    local_dir="models",
    allow_patterns=["hunyuan3d-delight-v2-0/**"]
)
```

**T5 NF4 encoder:**

```python
snapshot_download(
    repo_id="diffusers/t5-nf4",
    local_dir="models/t5-nf4",
    local_dir_use_symlinks=False
)
```

**Stable Diffusion 3.5 LT:**

```python
from huggingface_hub import login, snapshot_download

login(token="YOUR_HUGGINGFACE_TOKEN")

snapshot_download(
    repo_id="stabilityai/stable-diffusion-3.5-large-turbo",
    local_dir="models/stable-diffusion-3.5-large-turbo",
    local_dir_use_symlinks=False
)
```

---

## Usage

### Local Inference

- **Download the Models**
```bash
cd txt2_3D
python get_models.py
python get_img_models.py
```


- **2D Image Generation**

  1. Generate a 2D image from a text prompt using **Stable Diffusion 3.5 LT**.
  2. Optionally, use **FathomNet images** enhanced with SR_GAN and filtered for brightness, contrast, and sharpness.

- **3D Model Generation**

  1. Use **Hunyuan3D 2** for generating a 3D mesh from a single 2D image.
  2. Steps:
      - Generate a **mesh** from the image.
      - **Simplify the mesh** to reduce the number of triangles (from 10–15k to ~10%).
      - Generate **textured mesh** using the paint pipeline.
  3. Optional: Serve via **Flask endpoint** or visualize in web/Unity-based viewer.


Simply run

```bash
python main.py
```

### Web usage

to be filled by Dhruvanshu

### Unity usage

  to be filled by Shivansh

---

## Contributing

Contributions are what make the open-source community amazing. Any contributions you make are greatly appreciated.

1. Fork the Project  
2. Create your Feature Branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your Changes:
   ```bash
   git commit -m "Add some AmazingFeature"
   ```
4. Push to the Branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

---
## Future Goals

to be filled by Dhruvanshu

---
## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgements

to be filled by Dhruvanshu
