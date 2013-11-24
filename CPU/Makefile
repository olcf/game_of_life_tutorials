all: clean GOL_c GOL_f

GOL_c: GOL.c
	mkdir -p bin
	cc -o bin/GOL_c GOL.c
GOL_f: GOL.f90
	mkdir -p bin
	ftn -o bin/GOL_f GOL.f90
clean:
	rm -rf bin
