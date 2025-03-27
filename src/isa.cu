#include "isa.h"

void Result::print() const {
    if (resultKind == ResultKind::Address) {
        printf("[%d]", address);
    } else if (resultKind == ResultKind::Neighbour) {
        printf("neighbour");
    } else if (resultKind == ResultKind::External) {
        printf("external");
    }
};

void InputC::print() const {
    if (negated) {
        printf("~");
    }
    if (input.inputKind == InputKind::Address) {
        printf("[%d]", input.address);
    } else if (input.inputKind == InputKind::PD) {
        printf("PD");
    } else if (input.inputKind == InputKind::Up) {
        printf("up");
    } else if (input.inputKind == InputKind::Down) {
        printf("down");
    } else if (input.inputKind == InputKind::Right) {
        printf("right");
    } else if (input.inputKind == InputKind::Left) {
        printf("left");
    } else if (input.inputKind == InputKind::ZeroValue) {
        printf("0");
    }
};

void printCarry(Carry carry) {
    if (carry == Carry::Zero) {
        printf("0");
    } else if (carry == Carry::One) {
        printf("1");
    } else {
        printf("CR");
    }
}

void Instruction::print() const {
    if (isNop) {
        printf("_");
        return;
    }
    
    result.print();
    printf(" = ");
    input1.print();
    printf(" + ");
    input2.print();
    printf(" + ");
    printCarry(carry);
    printf(" ");
    printf("(%c)", resultType.value);
};

Program::Program(size_t vliwWidth, size_t count, Instruction* instr) 
    : vliwWidth(vliwWidth), instructionCount(count), instructions(instr) {}

void Program::print() const {
    for (size_t i = 0; i < instructionCount; ++i) {
        for (size_t j = 0; j < vliwWidth; ++j) {
            instructions[i * vliwWidth + j].print();
            if (j < vliwWidth - 1) {
                printf(" : ");
            }
        }
        printf("\n");
    }
}

Parser::Parser(const std::string &input): input(input), pos(0) {};

Program Parser::parse(size_t vliw) {
    std::vector<std::vector<Instruction>> instructionList;
    
    while (true) {
        skipWhitespace();
        if (match("end")) {
            break;
        }
        
        instructionList.push_back(parseVliwInstruction(vliw));
        expect(';');
    }
    
    Instruction* instructions = new Instruction[instructionList.size() * vliw];
    for (size_t i = 0; i < instructionList.size(); ++i) {
        std::copy(instructionList[i].begin(), instructionList[i].end(), instructions + i * vliw);
    }
    
    return Program(vliw, instructionList.size(), instructions);
}

std::vector<Instruction> Parser::parseVliwInstruction(size_t vliw) {
    std::vector<Instruction> vliwInstructions;
    
    for (size_t i = 0; i < vliw; ++i) {
        skipWhitespace();
        
        if (i == vliw - 1) {
            vliwInstructions.push_back(parseInstruction());
        } else {
            vliwInstructions.push_back(parseInstruction());
            expect(':');
        }
    }
    
    return vliwInstructions;
}

void Parser::skipWhitespace() {
    while (pos < input.size() && std::isspace(input[pos])) {
        ++pos;
    }
}

bool Parser::match(const std::string &str) {
    skipWhitespace();
    size_t start = pos;
    for (char ch : str) {
        if (pos >= input.size() || input[pos] != ch) {
            pos = start;
            return false;
        }
        ++pos;
    }
    return true;
}

void Parser::expect(char ch) {
    skipWhitespace();
    if (pos >= input.size() || input[pos] != ch) {
        throw std::runtime_error(std::string("Expected '") + ch + "' at position " + std::to_string(pos));
    }
    ++pos;
}

Result Parser::parseResult() {
    skipWhitespace();
    Result result;
    if (match("neighbour")) {
        result.resultKind = ResultKind::Neighbour;
    } else if (match("external")) {
        result.resultKind = ResultKind::External;
    } else if (match("[")) {
        result.resultKind = ResultKind::Address;
        result.address = std::stoi(parseNumber());
        expect(']');
    } else {
        throw std::runtime_error("Invalid result at position " + std::to_string(pos));
    }
    return result;
}

Input Parser::parseInput() {
    skipWhitespace();
    Input inputNode; 
    if (match("[")) {
        inputNode.inputKind = InputKind::Address;
        inputNode.address = std::stoi(parseNumber());
        expect(']');
    } else if (match("PD")) {
        inputNode.inputKind = InputKind::PD;
    } else if (match("up")) {
        inputNode.inputKind = InputKind::Up;
    } else if (match("down")) {
        inputNode.inputKind = InputKind::Down;
    } else if (match("right")) {
        inputNode.inputKind = InputKind::Right;
    } else if (match("left")) {
        inputNode.inputKind = InputKind::Left;
    } else if (match("0")) {
        inputNode.inputKind = InputKind::ZeroValue;
    } else {
        throw std::runtime_error("Invalid input at position " + std::to_string(pos));
    }
    return inputNode;
}

InputC Parser::parseInputC() {
    InputC inputC; 
    if (match("~")) {
        inputC.negated = true;
    }
    inputC.input = parseInput();
    return inputC;
}

Carry Parser::parseCarry() {
    skipWhitespace();
    Carry carry; 
    if (match("0")) {
        carry = Carry::Zero; // "0"
    } else if (match("1")) {
        carry = Carry::One; // "1"
    } else if (match("CR")) {
        carry = Carry::CR; // "CR"
    } else {
        throw std::runtime_error("Invalid carry at position " + std::to_string(pos));
    }
    return carry;
}

ResultType Parser::parseResultType() {
    expect('(');
    ResultType resultType; 
    if (match("s") || match("c")) {
        resultType.value = input[pos - 1]; // 's' or 'c'
    } else {
        throw std::runtime_error("Invalid result type at position " + std::to_string(pos));
    }
    expect(')');
    return resultType;
}

Instruction Parser::parseInstruction() {
    Instruction instruction;
    
    skipWhitespace();
    if (match("_")) {
        instruction.isNop = true;
        return instruction;
    }
    
    instruction.isNop = false;
    instruction.result = parseResult();
    expect('=');
    instruction.input1 = parseInputC();
    expect('+');
    instruction.input2 = parseInputC();
    expect('+');
    instruction.carry = parseCarry();
    instruction.resultType = parseResultType();
    return instruction;
}

std::string Parser::parseNumber() {
    skipWhitespace();
    size_t start = pos;
    while (pos < input.size() && std::isdigit(input[pos])) {
        ++pos;
    }
    if (start == pos) {
        throw std::runtime_error("Expected a number at position " + std::to_string(pos));
    }
    return input.substr(start, pos - start);
}
