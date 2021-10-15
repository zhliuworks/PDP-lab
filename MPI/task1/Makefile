all:
	mpicc -g -Wall -o round_trip_time_test round_trip_time_test.c
run:
	mpiexec -n 2 ./round_trip_time_test
clean:
	rm round_trip_time_test