

main: main.cu
	nvcc  main.cu -o main

run:
	nsys nvprof ./main
.PHONY: run

clean:
	rm *.sqlite
	rm *.nsys-rep
.PHONY: clean
