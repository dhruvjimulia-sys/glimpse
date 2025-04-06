#include "program_utils.h"

size_t numOutputs(Program program) {
    size_t num_outputs = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (program.instructions[i].result.resultKind == ResultKind::External && !program.instructions[i].isNop) {
            num_outputs++;
        }
    }
    return num_outputs;
}

size_t numSharedNeighbours(Program program) {
    size_t num_shared_neighbours = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (program.instructions[i].result.resultKind == ResultKind::Neighbour && !program.instructions[i].isNop) {
            num_shared_neighbours++;
        }
    }
    return num_shared_neighbours;
}

size_t numComputeAccesses(Program program) {
    size_t num_compute_accesses = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (!program.instructions[i].isNop) {
            num_compute_accesses++;
        }
    }
    return num_compute_accesses;
}