all:
	mpicc -g -Wall -o pass_string_problem pass_string_problem.c
run:
	mpiexec -n 5 ./pass_string_problem
clean:
	rm pass_string_problem