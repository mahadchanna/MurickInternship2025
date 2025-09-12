import math

try:
    num = int(input("Enter a number: "))

    if num > 0:
        print("The number is Positive.")
    elif num < 0:
        print("The number is Negative.")
    else:
        print("The number is Zero.")

    if num != 0:
        if num % 2 == 0:
            print("The number is Even.")
        else:
            print("The number is Odd.")

    if num >= 0:
        root = int(math.isqrt(num))
        if root * root == num:
            print("The number is a Perfect Square.")
        else:
            print("The number is NOT a Perfect Square.")

    if num > 0 and (num & (num - 1)) == 0:
        print("The number is a Power of 2.")
    else:
        if num > 0:
            print("The number is NOT a Power of 2.")

except ValueError:
    print("Error: Please enter a valid integer.")
