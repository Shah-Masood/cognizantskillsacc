age = int(input("How old are you? "))

if age >= 18:
    print("Congratulations! You are eligible to vote. Go make a difference!")
else :
    years = 18 - age
    print (f"Oops! You’re not eligible yet. But hey, only {years} more years to go")