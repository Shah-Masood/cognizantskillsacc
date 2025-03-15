inventory = {"apple": (10, 2.5),"banana": (20, 1.2)}

print("Welcome to the Inventory Manager!")
print("Current inventory:")
for item, (quantity, price) in inventory.items():
    print(f"Item: {item}, Quantity: {quantity}, Price: ${price}")

# Adding items
new_item = "mango"
new_quantity = 15
new_price = 3.0
print(f"Adding a new item: {new_item}")
inventory[new_item] = (new_quantity, new_price)

print("Updated inventory:")
for item, (quantity, price) in inventory.items():
    print(f"Item: {item}, Quantity: {quantity}, Price: ${price}")

total_value = sum(quantity * price for quantity, price in inventory.values())
print(f"Total value of inventory: ${total_value:.1f}")