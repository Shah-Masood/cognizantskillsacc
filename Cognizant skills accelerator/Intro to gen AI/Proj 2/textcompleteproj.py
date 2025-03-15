import openai

# Set your OpenAI API key
openai.api_key = "key"

def generate_text(prompt, max_length=100, temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,  # Controls creativity
            max_tokens=max_length
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# Interactive prompt loop
print("Welcome to AI Text Completion! Type 'exit' to quit.")
while True:
    user_input = input("\nEnter a prompt: ")
    if user_input.lower() == "exit":
        break
    result = generate_text(user_input)
    print("\nAI Response:\n", result)