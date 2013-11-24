all: checkEnv clean GOL_cl GOL_local_cl

.PHONY: checkEnv clean

GOL_cl: GOL.c
	mkdir -p bin
	cc -lOpenCL -o bin/GOL_cl GOL.c

GOL_local_cl: GOL_local.c
	mkdir -p bin
	CC -lOpenCL -o bin/GOL_local_cl GOL_local.c

checkEnv:
ifndef CRAY_CUDATOOLKIT_VERSION
	$(error cudatoolkit module not loaded)
endif

clean:
	rm -rf bin
