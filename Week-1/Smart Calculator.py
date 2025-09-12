try:
    num1 = float(input("Enter first number: "))
    operator = input("Enter operator (+, -, *, /, **, %): ")
    num2 = float(input("Enter second number: "))

    result = None

    if operator == "+":
        result = num1 + num2
    elif operator == "-":
        result = num1 - num2
    elif operator == "*":
        result = num1 * num2
    elif operator == "/":
        if num2 == 0:
            print("Error: Division by zero is not allowed.")
        else:
            result = num1 / num2
    elif operator == "**":
        result = num1 ** num2
    elif operator == "%":
        if num2 == 0:
            print("Error: Modulus by zero is not allowed.")
        else:
            result = num1 % num2
    else:
        print("Invalid operator!")

    if result is not None:
        print(f"{num1} {operator} {num2} = {result}")

except ValueError:
    print("Invalid input! Please enter numeric values.")
