

main: mwc_cuda.cu
	nvcc  mwc_cuda.cu -DMAIN=1 -o /tmp/main

run: main
	/tmp/main
.PHONY: run

profile:
	nsys nvprof /tmp/main
.PHONY: profile

clean:
	rm *.sqlite
	rm *.nsys-rep
.PHONY: clean
