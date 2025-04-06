#pragma once
#include "../isa.h"

size_t numOutputs(Program program);

size_t numSharedNeighbours(Program program);

size_t numComputeAccesses(Program program);