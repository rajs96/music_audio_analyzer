.PHONY: clean build build_amd push_image push_amd_image login

# Docker image settings
DOCKER_USER := rajs966
IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu
AMD_IMAGE_NAME := $(DOCKER_USER)/raj_audio_analyzer_gpu_amd
TS := $(shell date +%Y%m%d_%H%M%S)
FULL_TAG := $(IMAGE_NAME):$(TS)
AMD_FULL_TAG := $(AMD_IMAGE_NAME):$(TS)

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -type f -delete

login:
	docker login

push_image: login build
	docker push $(FULL_TAG)

push_amd_image: login build_amd
	docker push $(AMD_FULL_TAG)

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
