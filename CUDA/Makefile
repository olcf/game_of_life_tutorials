all: checkEnv clean GOL_cu GOL_shared_cu GOL_texture_cu

.PHONY: checkEnv clean

GOL_cu: GOL.cu
	mkdir -p bin
	nvcc -arch=sm_35 -o bin/GOL_cu GOL.cu

GOL_shared_cu: GOL_shared.cu
	mkdir -p bin
	nvcc -arch=sm_35 -o bin/GOL_shared_cu GOL_shared.cu

GOL_texture_cu: GOL_texture.cu
	mkdir -p bin
	nvcc -arch=sm_35 -o bin/GOL_texture_cu GOL_texture.cu

checkEnv:
ifndef CRAY_CUDATOOLKIT_VERSION
	$(error cudatoolkit module not loaded)
endif

clean:
	rm -rf bin
