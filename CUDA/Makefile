all:
	make baseline
	make opt_reduction
	make opt_squareddif
baseline:
	nvcc -o rbf0_baseline.out rbf0_baseline.cu
opt_reduction:
	nvcc -o rbf1_opt_reduction.out rbf1_opt_reduction.cu
opt_squareddif:
	nvcc -o rbf2_opt_squareddif.out rbf2_opt_squareddif.cu
clean:
	rm *.out