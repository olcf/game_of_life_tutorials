all: checkEnv clean GOL_cuf GOL_shared_cuf

.PHONY: checkEnv clean

GOL_cuf: GOL.cuf
	mkdir -p bin
	ftn -o bin/GOL_cuf GOL.cuf

GOL_shared_cuf: GOL_shared.cuf
	mkdir -p bin
	ftn -o bin/GOL_cuf GOL_shared.cuf
	

checkEnv:
ifndef CRAY_PRGENVPGI
	$(error PrgEnv-pgi module not loaded)
endif

ifndef CRAY_CUDATOOLKIT_VERSION
        $(error cudatoolkit module not loaded)
endif

clean:
	rm -rf bin
	rm -f kernels.mod
