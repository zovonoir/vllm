DOCKER_BUILDKIT=1 docker build . --build-arg PYTORCH_ROCM_ARCH=gfx950 --build-arg AITER_ROCM_ARCH=gfx950   -f docker/dockerfile.rocm_new -t sabreshao/vllm:aiter_0617
