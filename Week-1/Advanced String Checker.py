import string

def is_palindrome(s):

    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
    return cleaned == cleaned[::-1]

def is_alpha_only(s):

    cleaned = "".join(ch for ch in s if ch.isalpha())
    return cleaned != "" and cleaned.isalpha()

def are_parentheses_balanced(s):
    stack = []
    for ch in s:
        if ch == "(":
            stack.append(ch)
        elif ch == ")":
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0

text = input("Enter a string: ")

print("\n--- String Analysis ---")

if is_palindrome(text):
    print("The string IS a palindrome (ignoring spaces, punctuation, case).")
else:
    print("The string is NOT a palindrome.")

if is_alpha_only(text):
    print("The string contains only alphabetic characters (ignoring spaces/punctuation).")
else:
    print("The string contains non-alphabetic characters.")

if are_parentheses_balanced(text):
    print("Parentheses are balanced.")
else:
    print("Parentheses are NOT balanced.")
