#importing the random module that will help the program select a random word from the category chosen by the user.
import random

#ASCII art for hangman stages to make the game fun!
graphics = [
    #the "r" makes python treat the upcomin string as a raw string, which means it won't interpret the backslashes literally!
        r"""

     |/
     |
     |
     |
     |
     |
    """,
    r"""
          _______
     |/
     |
     |
     |
     |
     |
     """,
     r"""
           _______
     |/      |
     |
     |
     |
     |
     |
     """,
     r"""
           _______
     |/      |
     |      (_)
     |
     |
     |
     |
     """,
     r"""
           _______
     |/      |
     |      (_)
     |        |
     |
     |
     |
     """,
     r"""
           _______
     |/      |
     |      (_)
     |      \|
     |
     |
     |
     """,
     r"""
           _______
     |/      |
     |      (_)
     |      \|/
     |
     |
     |
     """,
     r"""
           _______
     |/      |
     |      (_)
     |      \|/
     |       |
     |
     |
     """,
     r"""
           _______
     |/      |
     |      (_)
     |      \|/
     |       |
     |      /
     |
     """,
     r"""
           _______
     |/      |
     |      (_)
     |      \|/
     |       |
     |      / \
     |
     """,
     r"""
           _______
     |/      |
     |      (x)
     |      \|/
     |       |
     |      / \
     |
     """
    ]

#making select() to enable the user to select a category of their choice.
def select():
    sets = {
        "fruits": ["mango", "banana", "orange", "apple", "plum"],
        "animals": ["horse", "rabbit", "elephant", "dolphin", "panther"],
        "colors": ["red", "yellow", "blue", "green", "white"]
    }

    #asking the user to enter a category out of fruits, animals, and colors and converting it to lowercase to remove case-sensitivity.
    print("Welcome to Hangman!")
    category = input("Choose a category from fruits/animals/colors: ").lower()

    #checking category is in the "sets" dictionary that we prescribed.
    while category not in sets:
        #re-prompting the user if it is not!
        print("Please enter a valid category")
        category = input("Choose a category from fruits/animals/colors: ").lower()

    #if the category is valid, the function select() should return a random word from the category.
    words = sets[category]
    return random.choice(words)

#defining the function play(), that takes two arguments, the randomly generated word (word), and the set of letters the user guesses (guessed)
def play(word, guessed):

    #initializing the value of display as " ", and then displaying every correctly guessed letter.
    display = " "
    for letter in word:
        if letter in guessed:
            display += letter
        else:
            display += " _ "
    return display

def main():

    #callling select() and generating a word from the user-selected category.
    word = select()
    #making sure the letters do not get repeated, using set() is a common practice!
    guessed = set()
    #setting maximum number of attempts.
    attempts = 10
    #initializing the incorrect number of guesses so the program can fetch the corresponding ascii art from the art list above.
    wrong = 0

    while wrong < 10:

        #displaying the appropriate hangman ascii art.
        print(graphics[wrong])
        #note that although we are literally printing the output of the play() function, the program shall only print " _ "s unless relevant letters are entered!
        print(f"\nWord: {play(word, guessed)}")
        #it is also important to convert the guessed letters to lowercase.
        guess = input("Guess a letter: ").lower()

        #making sure the user has entered ONE LETTER.
        if len(guess) != 1 or not guess.isalpha():
            print("Please enter a single letter!")
            continue

        #making sure the user has not re-guessed the same alphabet, the importance of set() used in line 42 is realized here!
        if guess in guessed:
            print("You've already guessed that letter!")
            continue

        #adding the newly guessed letter to the set of guessed letters.
        guessed.add(guess)

        #checking whether the guessed alphabet is in the secret word, afterall. If not, the program should decrease the number of remaining attempts.
        if guess not in word:
            wrong += 1
            print(f"Oops! '{guess}' is not in the word. {attempts - wrong} attempts left.")
        else:
            print(f"Good guess! '{guess}' is in the word.")

        #once the user has guessed the entire word perfectly, the program must prompt the user positively and end the program.
        if set(word).issubset(guessed):
            print(f"\nCongratulations! You guessed the word: {word}")
            break

    #if the user was unable to guess the word in 10 attempts, the program should reveal the word.
    if wrong == 10:
        print(f"\nGame over! The word was: {word}")


#execute the following code block only if this script is run directly (not when imported as a module) and call main.
if __name__ == "__main__":
    main()
