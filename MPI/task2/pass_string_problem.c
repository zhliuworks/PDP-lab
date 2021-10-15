#include <stdio.h>
#include <string.h>
#include <mpi.h>

const int NUM_CHILDS = 5;
const int NUM_STRING = 5;
const int MAX_STRING = 100;

int main(void) {
    char strings[NUM_STRING][MAX_STRING];
    char buffer[MAX_STRING];

    // send this to next child if recvd no words, as a label
    char none[MAX_STRING];

    int comm_sz;  // number of processes
    int my_rank;  // my process rank
    int i;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    strcpy(none, "$#&<");

    if (my_rank == 0) {  // master process
        strcpy(strings[0], "hello,world");  // child 2 no words recvd
        strcpy(strings[1], "bye,world");    // child 0 no words recvd
        strcpy(strings[2], "abcdefg");      // child 3 no words recvd
        strcpy(strings[3], "hello");        // child 1 no words recvd
        strcpy(strings[4], "abc");          // child 4 no words recvd

        for (i = 0; i < NUM_STRING; i++) {
            MPI_Send(strings[i], strlen(strings[i])+1, MPI_CHAR, 1, i, MPI_COMM_WORLD);
        }
    }

    // all processes
    for (i = 0; i < NUM_STRING; i++) {
        while (1) {
            MPI_Recv(buffer, MAX_STRING, MPI_CHAR, (my_rank-1+NUM_CHILDS)%NUM_CHILDS, 
                i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (!strlen(buffer)) {
                printf("# child %d received no word of string %d\n", my_rank, i);
                MPI_Send(none, strlen(none)+1, MPI_CHAR, (my_rank+1)%NUM_CHILDS, i, MPI_COMM_WORLD);
                break;                
            }
            if (!strcmp(buffer, none)) {
                MPI_Send(none, strlen(none)+1, MPI_CHAR, (my_rank+1)%NUM_CHILDS, i, MPI_COMM_WORLD);
                break;
            }
            printf("child %d received %c of string %d\n", my_rank, buffer[0], i);
            MPI_Send(buffer+1, strlen(buffer), MPI_CHAR, (my_rank+1)%NUM_CHILDS, i, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}