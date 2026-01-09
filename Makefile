.PHONY: clean clean_docker build build_amd push_image push_amd_image build_runpods build_amd_runpods push_runpods_image push_amd_runpods_image build_flash push_flash_image login login_ghcr build_vllm push_vllm_image build_vllm_h100 push_vllm_h100_image push_vllm_h100_ghcr_image build_vllm_flash push_vllm_flash_image push_vllm_flash_ghcr_image build_vllm_official push_vllm_official_image build_preprocessor push_preprocessor_image push_preprocessor_ghcr_image

# Docker image settings
DOCKER_USER := rajs966
GHCR_USER := rajs96
IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu
AMD_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_amd
RUNPODS_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_runpods
AMD_RUNPODS_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_runpods_amd
VLLM_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_vllm
VLLM_H100_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_vllm_h100
VLLM_OFFICIAL_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_vllm_official
PREPROCESSOR_IMAGE_NAME := $(DOCKER_USER)/raj_audio_preprocessor
VLLM_FLASH_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_vllm_flash
GHCR_VLLM_H100_IMAGE_NAME := ghcr.io/$(GHCR_USER)/raj_audio_analyzer_gpu_vllm_h100
GHCR_VLLM_FLASH_IMAGE_NAME := ghcr.io/$(GHCR_USER)/raj_audio_analyzer_gpu_vllm_flash
GHCR_PREPROCESSOR_IMAGE_NAME := ghcr.io/$(GHCR_USER)/raj_audio_preprocessor
TS := $(shell date +%Y%m%d_%H%M%S)
FULL_TAG := $(IMAGE_NAME):$(TS)
AMD_FULL_TAG := $(AMD_IMAGE_NAME):$(TS)
RUNPODS_FULL_TAG := $(RUNPODS_IMAGE_NAME):$(TS)
AMD_RUNPODS_FULL_TAG := $(AMD_RUNPODS_IMAGE_NAME):$(TS)
VLLM_FULL_TAG := $(VLLM_IMAGE_NAME):$(TS)
VLLM_H100_FULL_TAG := $(VLLM_H100_IMAGE_NAME):$(TS)
VLLM_OFFICIAL_FULL_TAG := $(VLLM_OFFICIAL_IMAGE_NAME):$(TS)
PREPROCESSOR_FULL_TAG := $(PREPROCESSOR_IMAGE_NAME):$(TS)
VLLM_FLASH_FULL_TAG := $(VLLM_FLASH_IMAGE_NAME):$(TS)
GHCR_VLLM_H100_FULL_TAG := $(GHCR_VLLM_H100_IMAGE_NAME):$(TS)
GHCR_VLLM_FLASH_FULL_TAG := $(GHCR_VLLM_FLASH_IMAGE_NAME):$(TS)
GHCR_PREPROCESSOR_FULL_TAG := $(GHCR_PREPROCESSOR_IMAGE_NAME):$(TS)

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -type f -delete

clean_docker:
	docker system prune -af
	docker rmi -f $$(docker images -q) 2>/dev/null || true

login:
	docker login

login_ghcr:
	docker login ghcr.io

push_image: login build
	docker push $(FULL_TAG)

push_amd_image: login build_amd
	docker push $(AMD_FULL_TAG)

push_runpods_image: login build_runpods
	docker push $(RUNPODS_FULL_TAG)

push_amd_runpods_image: login build_amd_runpods
	docker push $(AMD_RUNPODS_FULL_TAG)

build_amd:
	docker buildx build \
		--platform linux/amd64 \
		--no-cache \
		-f Dockerfile_gpu \
		-t $(AMD_FULL_TAG) \
		.

	@echo "Built image: $(AMD_FULL_TAG)"

build:
	docker buildx build \
		--platform linux/arm64 \
		--no-cache \
		-f Dockerfile_gpu \
		-t $(FULL_TAG) \
		.

	@echo "Built image: $(FULL_TAG)"

build_runpods:
	docker buildx build \
		--platform linux/amd64 \
		--no-cache \
		-f Dockerfile_gpu_runpods_v2 \
		-t $(RUNPODS_FULL_TAG) \
		.

	@echo "Built image: $(RUNPODS_FULL_TAG)"

build_amd_runpods:
	docker buildx build \
		--platform linux/amd64 \
		--no-cache \
		-f Dockerfile_gpu_runpods_v2_a100 \
		-t $(AMD_RUNPODS_FULL_TAG) \
		.

	@echo "Built image: $(AMD_RUNPODS_FULL_TAG)"

build_vllm_h100:
	docker buildx build \
		--platform linux/amd64 \
		--no-cache \
		-f Dockerfile_vllm_h100 \
		-t $(VLLM_H100_FULL_TAG) \
		.

	@echo "Built image: $(VLLM_H100_FULL_TAG)"

push_vllm_h100_image: login build_vllm_h100
	docker push $(VLLM_H100_FULL_TAG)

push_vllm_h100_ghcr_image: login_ghcr build_vllm_h100
	docker tag $(VLLM_H100_FULL_TAG) $(GHCR_VLLM_H100_FULL_TAG)
	docker push $(GHCR_VLLM_H100_FULL_TAG)
	@echo "Pushed image: $(GHCR_VLLM_H100_FULL_TAG)"


format:
	black .
