

main: mwc_cuda.cu
	nvcc  mwc_cuda.cu -DMAIN=1 -o /tmp/main

run:
	nsys nvprof /tmp/main
.PHONY: run

clean:
	rm *.sqlite
	rm *.nsys-rep
.PHONY: clean
