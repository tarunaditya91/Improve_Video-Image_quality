# Improve_Video-Image_quality
CodeFormer Setup and Usage Guide
Prerequisites
Ensure you have the following software installed on your system:

Anaconda or Miniconda
Git
NVIDIA CUDA Toolkit (if using a GPU)
Step-by-Step Installation
1. Create and Activate the Conda Environment
Open your terminal (or Anaconda Prompt) and create a new conda environment named Codeformer3 with Python 3.10:

bash
Copy code
conda create --name Codeformer3 python=3.10
Activate the environment:

bash
Copy code
conda activate Codeformer3
2. Clone the CodeFormer Repository
Clone the CodeFormer repository from GitHub:

bash
Copy code
git clone https://github.com/sczhou/CodeFormer.git
cd CodeFormer
3. Check CUDA Installation
Verify your CUDA installation:

bash
Copy code
nvcc --version
4. Install PyTorch with CUDA Support
Install PyTorch with CUDA support (ensure the version matches your CUDA toolkit):

bash
Copy code
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
5. Install Required Python Packages
Install the required Python packages listed in requirements.txt:

bash
Copy code
pip install addict future lmdb numpy opencv-python Pillow==9.5.0 requests scikit-image scipy tb-nightly tqdm yapf lpips gdown
6. Install Additional Dependencies
Install additional dependencies:

bash
Copy code
conda install -c conda-forge dlib
7. Set Up the CodeFormer Project
Run the setup script for basicsr:

bash
Copy code
python basicsr/setup.py develop
8. Download Pre-trained Models
Download the necessary pre-trained models:

bash
Copy code
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py dlib
python scripts/download_pretrained_models.py CodeFormer
9. Install FFmpeg
Ensure ffmpeg is installed and accessible:

bash
Copy code
conda install -c conda-forge ffmpeg
10. Run CodeFormer Inference
Run the inference on an input image:

bash
Copy code
python inference_codeformer.py --input_path inputs/whole_imgs --output_path results -w 0.7 --bg_upsampler realesrgan --face_upsample
Troubleshooting
1. ffmpeg Module Not Found
If you encounter an issue with the ffmpeg module not being found:

Verify the installation of ffmpeg-python:

bash
Copy code
pip show ffmpeg-python
Install imageio-ffmpeg:

bash
Copy code
pip install imageio-ffmpeg
Check if ffmpeg is in your system's PATH:

bash
Copy code
where ffmpeg
Update ffmpeg-python:

bash
Copy code
pip install --upgrade ffmpeg-python
Modify video_util.py to use imageio as a fallback:

python
Copy code
try:
    import ffmpeg
except ModuleNotFoundError:
    import imageio_ffmpeg as ffmpeg
2. Running Inference on Videos
Ensure you have ffmpeg-python installed and try running the inference on a video file:

bash
Copy code
python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path video/Startrek.avi
Conclusion
You have successfully set up the CodeFormer project and run inference on both images and videos. For any additional issues or specific error messages, consult the project's GitHub issues page or the community forums.



![image](https://github.com/tarunaditya91/Improve_Video-Image_quality/assets/113850656/1434c571-b84a-4601-94b8-3edb31082f35)




