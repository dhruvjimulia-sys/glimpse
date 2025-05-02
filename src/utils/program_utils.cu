#include "program_utils.h"
#include <unordered_set>
#include "../isa.h"

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

size_t numMemoryReadAccesses(Program program) {
    size_t num_memory_read_accesses = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (!program.instructions[i].isNop) {
            if (program.instructions[i].input1.input.inputKind == InputKind::Address) {
                num_memory_read_accesses++;
            }
            if (program.instructions[i].input2.input.inputKind == InputKind::Address) {
                num_memory_read_accesses++;
            }
        }
    }
    return num_memory_read_accesses;
}

size_t numMemoryWriteAccesses(Program program) {
    size_t num_memory_write_accesses = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (!program.instructions[i].isNop) {
            if (program.instructions[i].result.resultKind == ResultKind::Address) {
                num_memory_write_accesses++;
            }
        }
    }
    return num_memory_write_accesses;
}

size_t numRegisterReadAccesses(Program program) {
    size_t num_register_read_accesses = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (!program.instructions[i].isNop) {
            if (program.instructions[i].carry == Carry::CR) {
                num_register_read_accesses += 4;
            } else {
                num_register_read_accesses += 3;
            }
        }
    }
    return num_register_read_accesses;
}

size_t numRegisterWriteAccesses(Program program) {
    // number of writes to internal registers turns out to be the same as number of reads
    return numRegisterReadAccesses(program);
}

double utilization(Program program) {
    double num_nops = 0;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (program.instructions[i].isNop) {
            num_nops++;
        }
    }
    double num_instructions = program.instructionCount * program.vliwWidth;
    double num_active_instructions = num_instructions - num_nops;
    double utilization = num_active_instructions / num_instructions;
    return utilization;
}

size_t memoryUsage(Program program) {
    std::unordered_set<int> memory_addresses;
    for (int i = 0; i < program.instructionCount * program.vliwWidth; i++) {
        if (!program.instructions[i].isNop) {
            if (program.instructions[i].input1.input.inputKind == InputKind::Address) {
                memory_addresses.insert(program.instructions[i].input1.input.address);
            }
            if (program.instructions[i].input2.input.inputKind == InputKind::Address) {
                memory_addresses.insert(program.instructions[i].input2.input.address);
            }
            if (program.instructions[i].result.resultKind == ResultKind::Address) {
                memory_addresses.insert(program.instructions[i].result.address);
            }
        }
    }
    size_t memory_usage = memory_addresses.size();
    return memory_usage;
}