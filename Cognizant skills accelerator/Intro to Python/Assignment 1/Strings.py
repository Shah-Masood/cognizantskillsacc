#task 1

sentence = "Python is Amazing"

#print python
print("First Word: " + sentence[:6])
#print amazing
print("Last Word: " + sentence[10:])
#print reverse
print("Reversed: " + sentence[::-1])

#task 2

sentence2 = " hello, python world! "

#remove space
print(sentence2.strip())
#Capitalize first letter
print(sentence2.capitalize())
#replace word
print(sentence2.replace("world", "universe"))
#replace it with uppercase
print(sentence2.upper())

#task 3

instring = input("Enter a String: ")

if instring == instring[::-1]:
    print("Yep its a palindrome")
else:
    print("Not a Palindrome")
