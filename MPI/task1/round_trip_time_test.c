#include <stdio.h>
#include <string.h>
#include <mpi.h>

const int NUM_BEACON_PACKETS = 10;
const int MAX_STRING = 100;

int main(void) {
    char beaconPacket[MAX_STRING];
    char ackPacket[MAX_STRING];
    char recvBuffer[MAX_STRING];  // data receive buffer
    char refBuffer[MAX_STRING]; // reference buffer
    int comm_sz;  // number of processes
    int my_rank;  // my process rank
    int i;

    double MPI_Wtime(void);
    double start[NUM_BEACON_PACKETS];  // round trip start time
    double end[NUM_BEACON_PACKETS];  // round trip end time

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {  // master process (task 0)
        // send packet to worker process(1) with tag i
        for (i = 0; i < NUM_BEACON_PACKETS; i++) {
            sprintf(beaconPacket, "0: sent beacon packet %d", i);
            start[i] = MPI_Wtime();
            MPI_Send(beaconPacket, strlen(beaconPacket)+1, MPI_CHAR, 1, i, MPI_COMM_WORLD);
        }
        // try to receive packet with tag i
        for (i = 0; i < NUM_BEACON_PACKETS; i++) {
            MPI_Recv(recvBuffer, MAX_STRING, MPI_CHAR, 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            end[i] = MPI_Wtime();
            sprintf(refBuffer, "1: sent ack packet %d", i);
            if (!strcmp(recvBuffer, refBuffer)) {
                printf("0: received ack packet %d in %f seconds\n", i, end[i]-start[i]);
            } else {
                printf("0: NOT received ack packet %d !!!\n", i);
            }
        }
    } else {  // worker process (task 1)
        // try to receive packet with tag i and send ack
        for (i = 0; i < NUM_BEACON_PACKETS; i++) {
            MPI_Recv(recvBuffer, MAX_STRING, MPI_CHAR, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sprintf(refBuffer, "0: sent beacon packet %d", i);
            if (!strcmp(recvBuffer, refBuffer)) {
                printf("1: received beacon packet %d\n", i);
                sprintf(ackPacket, "1: sent ack packet %d", i);
                MPI_Send(ackPacket, strlen(ackPacket)+1, MPI_CHAR, 0, i, MPI_COMM_WORLD);
            } else {
                printf("1: NOT received beacon packet %d !!!\n", i);
            }
        }
    }

    MPI_Finalize();
    return 0;
}