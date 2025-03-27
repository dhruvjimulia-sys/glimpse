#include "program_utils.h"

size_t numOutputs(Program program) {
    size_t num_outputs = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (program.instructions[i].result.resultKind == ResultKind::External) {
            num_outputs++;
        }
    }
    return num_outputs;
}

size_t numSharedNeighbours(Program program) {
    size_t num_shared_neighbours = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (program.instructions[i].result.resultKind == ResultKind::Neighbour) {
            num_shared_neighbours++;
        }
    }
    return num_shared_neighbours;
}