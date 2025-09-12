import math

def fibonacci(limit):
    seq = []
    a, b = 0, 1
    while a <= limit:
        seq.append(a)
        a, b = b, a + b
    return seq

def primes(limit):
    seq = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            seq.append(num)
    return seq

def perfect_squares(limit):
    seq = []
    n = 1
    while n * n <= limit:
        seq.append(n * n)
        n += 1
    return seq

def main():
    print("\n Sequence Generator ")
    print("1. Fibonacci sequence")
    print("2. Prime numbers")
    print("3. Perfect squares")
    print("4. Fibonacci primes (both Fibonacci & prime)")
    print("5. Prime squares (prime numbers that are also perfect squares — only 4)")

    choice = input("\nEnter your choice (1–5): ").strip()

    try:
        limit = int(input("Enter the limit: "))
    except ValueError:
        print("Invalid input! Limit must be a number.")
        return

    if choice == "1":
        print("\nFibonacci sequence up to", limit, ":\n", fibonacci(limit))
    elif choice == "2":
        print("\nPrime numbers up to", limit, ":\n", primes(limit))
    elif choice == "3":
        print("\nPerfect squares up to", limit, ":\n", perfect_squares(limit))
    elif choice == "4":
        fib = set(fibonacci(limit))
        prime = set(primes(limit))
        fib_primes = sorted(list(fib & prime))
        print("\nFibonacci primes up to", limit, ":\n", fib_primes)
    elif choice == "5":
        prime = set(primes(limit))
        squares = set(perfect_squares(limit))
        prime_squares = sorted(list(prime & squares))
        print("\nPrime perfect squares up to", limit, ":\n", prime_squares)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
