#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cctype>

// AST node classes
struct ASTNode {
    virtual ~ASTNode() = default;
};

struct Input : ASTNode {
    std::string value;
};

struct ResultType : ASTNode {
    char value; // 's' or 'c'
};

struct Result : ASTNode {
    std::string value;
};

struct InputC : ASTNode {
    bool negated = false;
    std::shared_ptr<Input> input;
};

struct Carry : ASTNode {
    std::string value; // "0", "1", or "CR"
};

struct Instruction : ASTNode {
    std::shared_ptr<Result> result;
    std::shared_ptr<InputC> input1;
    std::shared_ptr<InputC> input2;
    std::shared_ptr<Carry> carry;
    std::shared_ptr<ResultType> resultType;

    void print() const {
        std::cout << "Result: " << result->value << ", Input1: " << input1->input->value
                  << ", Input2: " << input2->input->value << ", Carry: " << carry->value
                  << ", ResultType: " << resultType->value << std::endl;
    }
};

struct Program : ASTNode {
    std::vector<std::shared_ptr<Instruction>> instructions;

    void print() const {
        for (const auto& instruction : instructions) {
            instruction->print();
        }
    }
};

// Parser
class Parser {
public:
    explicit Parser(const std::string &input) : input(input), pos(0) {}

    std::shared_ptr<Program> parse() {
        auto program = std::make_shared<Program>();
        while (true) {
            skipWhitespace();
            if (match("end")) {
                break;
            }
            program->instructions.push_back(parseInstruction());
            expect(';');
        }
        return nullptr;
    }

private:
    std::string input;
    size_t pos;

    void skipWhitespace() {
        while (pos < input.size() && std::isspace(input[pos])) {
            ++pos;
        }
    }

    bool match(const std::string &str) {
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

    void expect(char ch) {
        skipWhitespace();
        if (pos >= input.size() || input[pos] != ch) {
            throw std::runtime_error(std::string("Expected '") + ch + "' at position " + std::to_string(pos));
        }
        ++pos;
    }

    std::shared_ptr<Result> parseResult() {
        skipWhitespace();
        auto result = std::make_shared<Result>();
        if (match("neighbour") || match("external")) {
            result->value = input.substr(pos - 9, 9); // "neighbour" or "external"
        } else if (match("[")) {
            result->value = "[" + parseNumber() + "]";
            expect(']');
        } else {
            throw std::runtime_error("Invalid result at position " + std::to_string(pos));
        }
        return result;
    }

    std::shared_ptr<Input> parseInput() {
        skipWhitespace();
        auto inputNode = std::make_shared<Input>();
        if (match("[")) {
            inputNode->value = "[" + parseNumber() + "]";
            expect(']');
        } else {
            static const std::vector<std::string> keywords = {"PD", "up", "down", "right", "left", "0"};
            for (const auto &kw : keywords) {
                if (match(kw)) {
                    inputNode->value = kw;
                    return inputNode;
                }
            }
            throw std::runtime_error("Invalid input at position " + std::to_string(pos));
        }
        return inputNode;
    }

    std::shared_ptr<InputC> parseInputC() {
        auto inputC = std::make_shared<InputC>();
        if (match("~")) {
            inputC->negated = true;
        }
        inputC->input = parseInput();
        return inputC;
    }

    std::shared_ptr<Carry> parseCarry() {
        skipWhitespace();
        auto carry = std::make_shared<Carry>();
        if (match("0") || match("1") || match("CR")) {
            carry->value = input.substr(pos - 1, 1); // "0" or "1" or "CR"
        } else {
            throw std::runtime_error("Invalid carry at position " + std::to_string(pos));
        }
        return carry;
    }

    std::shared_ptr<ResultType> parseResultType() {
        expect('(');
        auto resultType = std::make_shared<ResultType>();
        if (match("s") || match("c")) {
            resultType->value = input[pos - 1]; // 's' or 'c'
        } else {
            throw std::runtime_error("Invalid result type at position " + std::to_string(pos));
        }
        expect(')');
        return resultType;
    }

    std::shared_ptr<Instruction> parseInstruction() {
        auto instruction = std::make_shared<Instruction>();
        instruction->result = parseResult();
        expect('=');
        instruction->input1 = parseInputC();
        expect('+');
        instruction->input2 = parseInputC();
        expect('+');
        instruction->carry = parseCarry();
        instruction->resultType = parseResultType();
        return instruction;
    }

    std::string parseNumber() {
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
};

// Test the parser
// int main() {
//     std::string programText =
//         "[3] = [2] + ~PD + 1 (s);"
//         "neighbour = PD + down + CR (c);"
//         "end";

//     try {
//         Parser parser(programText);
//         std::shared_ptr<Program> program = parser.parse();
//         program->print();
//         std::cout << "Program parsed successfully!" << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }

//     return 0;
// }
