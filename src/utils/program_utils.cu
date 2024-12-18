#include "program_utils.h"

size_t numOutputs(Program program) {
    size_t num_outputs = 0;
    for (int i = 0; i < program.instructionCount; i++) {
        if (program.instructions[i].result.resultKind == ResultKind::External) {
            num_outputs++;
        }
    }
    return num_outputs;
}