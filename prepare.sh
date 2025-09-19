conda create -n rekv 
conda init

# restart
conda activate rekv 
# conda activate rekvbash prepare.sh

apt-get update
apt-get install -y git vim tmux ffmpeg wget curl rsync git-lfs

pip install transformers==4.52.0 qwen_vl_utils
pip install accelerate pillow matplotlib bitsandbytes
pip install decord opencv-python tqdm datasets
pip install flash-attn --no-build-isolation
# --no-build-isolation = 내 현재 환경을 그대로 써서 빌드 (PyTorch/CUDA 확장 빌드 시 필수)


# /workspace/ReKV/
pip install -U "git+https://github.com/huggingface/transformers.git@66bc4def9505fa7c7fe4aa7a248c34a026bb552b"
pip install -e .

cd model/longva/
pip install -e .

