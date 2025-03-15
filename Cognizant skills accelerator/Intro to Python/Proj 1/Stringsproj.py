import string

password = input("Enter New Password: ")

length = len(password)
lower = any(c.islower() for c in password)
upper = any(c.isupper() for c in password)
digit = any(c.isdigit() for c in password)
special = any(c in string.punctuation for c in password)

score = sum([length >= 8, lower, upper, digit, special])

if score == 5:
    print("Your password is strong!")
else:
    if length < 8:
        print("Your password needs to be at least 8 characters long.")
    if not lower:
        print("Your password needs at least one lowercase letter.")
    if not upper:
        print("Your password needs at least one uppercase letter.")
    if not digit:
        print("Your password needs at least one digit.")
    if not special:
        print("Your password needs at least one special character.")