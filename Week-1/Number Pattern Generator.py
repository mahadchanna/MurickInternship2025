try:
    start = int(input("Enter start of range: "))
    end = int(input("Enter end of range: "))

    divisor1 = int(input("Enter first divisor: "))
    word1 = input("Enter word for first divisor: ")

    divisor2 = int(input("Enter second divisor: "))
    word2 = input("Enter word for second divisor: ")

    print("\n--- Output ---")
    for num in range(start, end + 1):
        output = ""
        if num % divisor1 == 0:
            output += word1
        if num % divisor2 == 0:
            output += word2

        print(output if output else num)

except ValueError:
    print("Error: Please enter valid numbers for range and divisors.")
