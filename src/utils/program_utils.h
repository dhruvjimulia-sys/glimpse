#pragma once
#include "../isa.h"

size_t numOutputs(Program program);

size_t numSharedNeighbours(Program program);

size_t numComputeAccesses(Program program);

size_t numMemoryReadAccesses(Program program);

size_t numMemoryWriteAccesses(Program program);

size_t numRegisterReadAccesses(Program program);

size_t numRegisterWriteAccesses(Program program);

double utilization(Program program);

size_t memoryUsage(Program program);