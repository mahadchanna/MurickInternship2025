import random

def get_hint(guess, secret):
    diff = abs(guess - secret)
    if guess > secret:
        if diff > 10:
            return "Much too high!"
        elif diff > 5:
            return "Close! A bit high."
        else:
            return "Very close! Just a little high."
    else:
        if diff > 10:
            return "Much too low!"
        elif diff > 5:
            return "Close! A bit low."
        else:
            return "Very close! Just a little low."

def play_game():
    print("\n🎮 Welcome to the Number Guessing Game! 🎮")


    print("\nChoose difficulty:")
    print("1. Easy (Unlimited attempts)")
    print("2. Medium (10 attempts)")
    print("3. Hard (5 attempts)")

    while True:
        try:
            choice = int(input("Enter difficulty level (1/2/3): "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Please choose 1, 2, or 3.")
        except ValueError:
            print("Invalid input, enter a number (1/2/3).")

    if choice == 1:
        attempts_allowed = None
    elif choice == 2:
        attempts_allowed = 10
    else:
        attempts_allowed = 5


    try:
        start = int(input("\nEnter start of range: "))
        end = int(input("Enter end of range: "))
    except ValueError:
        print("Invalid input! Using default range 1 to 100.")
        start, end = 1, 100

    secret = random.randint(start, end)
    attempts = 0
    score = 100

    print(f"\nI have picked a number between {start} and {end}. Try to guess it!\n")

    while True:
        try:
            guess = int(input("Your guess: "))
            attempts += 1

            if guess == secret:
                print(f"🎉 Correct! The number was {secret}.")
                print(f"Attempts: {attempts}")
                print(f"Your score: {score}")
                break
            else:
                print(get_hint(guess, secret))
                score -= 10

                if attempts_allowed and attempts >= attempts_allowed:
                    print(f"\n❌ Game Over! You used all {attempts_allowed} attempts.")
                    print(f"The correct number was {secret}.")
                    break

        except ValueError:
            print("Please enter a valid number.")


    replay = input("\nDo you want to play again? (y/n): ").strip().lower()
    if replay == "y":
        play_game()
    else:
        print("\nThanks for playing!")

if __name__ == "__main__":
    play_game()
