if [ -z "$1" ]; then
  echo "Usage: $0 <gpu_id>"
  exit 1
fi
CUDA_VISIBLE_DEVICES=$1 python inference_story.py