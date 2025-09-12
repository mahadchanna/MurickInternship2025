from collections import Counter
import string

sentence = input("Enter a sentence: ")

sentence = sentence.strip()

total_chars = len(sentence)

words = sentence.split()
word_count = len(words)

vowels = "aeiou"
vowel_count = 0
consonant_count = 0
for ch in sentence.lower():
    if ch.isalpha():
        if ch in vowels:
            vowel_count += 1
        else:
            consonant_count += 1

longest_word = max(words, key=len) if words else ""

letters_only = [ch.lower() for ch in sentence if ch.isalpha()]
frequency = Counter(letters_only)

print("\n--- Sentence Analysis Report ---")
print(f"Total characters (with spaces): {total_chars}")
print(f"Total words: {word_count}")
print(f"Vowels: {vowel_count}")
print(f"Consonants: {consonant_count}")
print(f"Longest word: {longest_word}")

print("\nLetter Frequencies:")
for letter, count in sorted(frequency.items()):
    print(f"{letter}: {count}")
