from collections import Counter

def are_anagrams(s1, s2):
    return Counter(s1.lower().replace(" ", "")) == Counter(s2.lower().replace(" ", ""))

def find_anagrams(word, word_list):
    word_count = Counter(word.lower())
    return [w for w in word_list if Counter(w.lower()) == word_count]

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1].lower() == s2[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],
                                   dp[i][j - 1],
                                   dp[i - 1][j - 1])
    return dp[m][n]

def spell_check(word, dictionary):

    return sorted(dictionary, key=lambda w: edit_distance(word, w))[:3]

def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + s1[i]
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], key=len)

    return dp[m][n]


if __name__ == "__main__":

    print("Anagram Check:")
    print(are_anagrams("listen", "silent"))
    print(are_anagrams("hello", "world"))

    word_list = ["enlist", "google", "inlets", "banana", "listen"]
    print("\nFind Anagrams:")
    print(find_anagrams("listen", word_list))

    dictionary = ["apple", "banana", "orange", "grape", "grapefruit"]
    print("\nSpell Check:")
    print(spell_check("appl", dictionary))

    print("\nLongest Common Subsequence:")
    print(longest_common_subsequence("programming", "gaming"))
