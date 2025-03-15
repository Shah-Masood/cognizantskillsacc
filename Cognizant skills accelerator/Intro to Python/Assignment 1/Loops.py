#task 1

num = int(input("Enter a number to start the countdown: "))

while num>=1:
    print (num); num = num - 1

print("Blast Off!")

#task2

num2 = int(input("Enter a number: "))

for i in range (1, 10):
    print(f"{num2} x " + str(i) + " = " + str(num2 * i))

#task 3

num3 = int(input("Enter a number: "))

for i in range(1, num3):
    num3 = num3 * i

print("Factorial of given number is " + str(num3))