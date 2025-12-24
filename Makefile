.PHONY: clean build build_amd push_image push_amd_image build_runpods build_amd_runpods push_runpods_image push_amd_runpods_image login

# Docker image settings
DOCKER_USER := rajs966
IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu
AMD_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_amd
RUNPODS_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_runpods
AMD_RUNPODS_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_runpods_amd
TS := $(shell date +%Y%m%d_%H%M%S)
FULL_TAG := $(IMAGE_NAME):$(TS)
AMD_FULL_TAG := $(AMD_IMAGE_NAME):$(TS)
RUNPODS_FULL_TAG := $(RUNPODS_IMAGE_NAME):$(TS)
AMD_RUNPODS_FULL_TAG := $(AMD_RUNPODS_IMAGE_NAME):$(TS)

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -type f -delete

login:
	docker login

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
		-f Dockerfile_gpu_runpods_v2 \
		-t $(AMD_RUNPODS_FULL_TAG) \
		.

	@echo "Built image: $(AMD_RUNPODS_FULL_TAG)"

format:
	black .
