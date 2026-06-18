DOCKER_BUILDKIT=1 docker build . \
  --build-arg PYTORCH_ROCM_ARCH=gfx950 \
  --build-arg AITER_ROCM_ARCH=gfx950 \
  --build-arg REMOTE_VLLM=1 \
  --build-arg VLLM_REPO=https://github.com/vllm-project/vllm.git \
  --build-arg VLLM_BRANCH=b4092176b9bc76839f76b0e91972a52e8b14ea2b \
  -f docker/dockerfile.rocm_new \
  -t sabreshao/vllm:aiter_0618
