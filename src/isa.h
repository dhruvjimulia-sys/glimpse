#pragma once
#include <cctype>
#include <iostream>
#include <string>
#include <vector>
// Note: MEMORY_SIZE_IN_BITS
#define MEMORY_SIZE_IN_BITS 130
#define CLOCK_FREQUENCY 2e8  // 200 MHz

enum class InputKind { Address, PD, Up, Down, Right, Left, ZeroValue };

struct Input {
    InputKind inputKind;
    uint8_t address;
};

// convert to enum
struct ResultType {
    char value;  // 's' or 'c'
};

enum class ResultKind { Address, Neighbour, External };

struct Result {
    ResultKind resultKind;
    uint8_t address;

    __device__ __host__ void print() const;
};

struct InputC {
    bool negated = false;
    Input input;

    __device__ __host__ void print() const;
};

enum class Carry { Zero, One, CR };

__device__ __host__ void printCarry(Carry carry);

struct Instruction {
    Result result;
    InputC input1;
    InputC input2;
    Carry carry;
    ResultType resultType;
    bool isNop;

    __device__ __host__ void print() const;
};

struct Program {
    Instruction* instructions;
    size_t instructionCount;
    size_t vliwWidth;
    bool isPipelining;

    Program(size_t vliwWidth, size_t count, Instruction* instr,
            bool isPipelining);
    __device__ __host__ void print() const;
};

class Parser {
   public:
    explicit Parser(const std::string& input);
    Program parse(size_t vliw, bool isPipelining);

   private:
    std::string input;
    size_t pos;

    void skipWhitespace();
    std::vector<Instruction> parseVliwInstruction(size_t vliw);
    bool match(const std::string& str);
    void expect(char ch);
    Result parseResult();
    Input parseInput();
    InputC parseInputC();
    Carry parseCarry();
    ResultType parseResultType();
    Instruction parseInstruction();
    std::string parseNumber();
};