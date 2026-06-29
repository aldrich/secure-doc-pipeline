import ollama

try:
    response = ollama.chat(
        model='llama3.2:3b', 
        messages=[
            {
                'role': 'user',
                'content': 'Respond with exactly two words: Its Working!'
            }
        ]
    )
    print("\n🚀 Connection successful!")
    print(f"Model response: {response['message']['content'].strip()}\n")
except Exception as e:
    print(f"\n❌ Connection failed. Is Ollama running in the background?")
    print(f"Error: {e}\n")