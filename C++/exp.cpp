#include <iostream>
#include <stack>
#include <string>

using namespace std;

bool solveBooleanExpression(string expression) {
    stack<bool> operands;
    stack<char> operators;

    for (int i = 0; i < expression.length(); i++) {
        if (expression[i] == 'T') {
            operands.push(true);
        } else if (expression[i] == 'F') {
            operands.push(false);
        } else if (expression[i] == '(') {
            operators.push(expression[i]);
        } else if (expression[i] == ')') {
            while (!operators.empty() && operators.top() != '(') {
                bool operand1 = operands.top();
                operands.pop();
                bool operand2 = operands.top();
                operands.pop();
                char op = operators.top();
                operators.pop();
                if (op == '&') {
                    operands.push(operand1 && operand2);
                } else if (op == '|') {
                    operands.push(operand1 || operand2);
                }
            }
            operators.pop();
        } else if (expression[i] == '&' || expression[i] == '|') {
            while (!operators.empty() && operators.top() != '(') {
                bool operand1 = operands.top();
                operands.pop();
                bool operand2 = operands.top();
                operands.pop();
                char op = operators.top();
                operators.pop();
                if (op == '&') {
                    operands.push(operand1 && operand2);
                } else if (op == '|') {
                    operands.push(operand1 || operand2);
                }
            }
            operators.push(expression[i]);
        }
    }

    while (!operators.empty()) {
        bool operand1 = operands.top();
        operands.pop();
        bool operand2 = operands.top();
        operands.pop();
        char op = operators.top();
        operators.pop();
        if (op == '&') {
            operands.push(operand1 && operand2);
        } else if (op == '|') {
            operands.push(operand1 || operand2);
        }
    }

    return operands.top();
}

int main() {
    string exp = "T&(F|T)";
    cout<<solveBooleanExpression(exp)<<endl;
    
    return 0;
}
