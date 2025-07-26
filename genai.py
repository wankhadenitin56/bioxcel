import google.generativeai as genai
API_KEY = "AIzaSyDX3ZKf5X9U_m2zLYzNVxNh1OCKAWEriUc"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()
print("Chat started with model:")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break
    chat.send_message(user_input)
    response = chat.send_message(user_input)
    print("Response:", response.text)

