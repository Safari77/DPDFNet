python 3.12+ is not supported.
Install python 3.11 first. On Fedora 42:
dnf5 install python3.11

then as user:
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip uninstall -y numpy
pip install "numpy<2"
pip install tflite-runtime librosa soundfile numpy tqdm
