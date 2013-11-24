all: clean checkEnv setFlags GOL_c GOL_f

.PHONY: checkEnv setFlags clean

checkenv:
ifdef CRAY_PRGENVCRAY
    ifndef HMPP_BIN_PATH
        ifndef CRAY_ACCEL_TARGET
            $(error craype-accel-nvidia35 not loaded)
        endif
    else
        ifdef CRAY_ACCEL_TARGET
            $(error craype-accel-nvidia35 not compatible with CapsMC. Use cudatoolkit module)
        endif
    endif
endif
ifndef CRAY_CUDATOOLKIT_VERSION
    $(error cudatoolkit module not loaded)
endif

setFlags:
ifdef HMPP_BIN_PATH
    ifdef CRAY_PRGENVGNU
        CFLAGS = -std=c99
    endif
    ifdef CRAY_PRGENVINTEL
        CFLAGS = -restrict
    endif
else
    ifdef CRAY_PRGENVPGI
        CFLAGS = -acc -Minline -ta=nvidia:kepler
        FFLAGS = -acc -Minline -ta=nvidia:kepler
    endif
    ifdef CRAY_PRGENVCRAY
        CFLAGS = -hpragma=acc
        FFLAGS = -hacc
    endif
endif

GOL_c: GOL.c
	mkdir -p bin
	cc $(CFLAGS) -o bin/GOL_c GOL.c
GOL_f: GOL.f90
	mkdir -p bin
	ftn $(FFLAGS) -o bin/GOL_f GOL.f90

clean:
	rm -rf bin
