# Tool call example using ollama (example from ollama.com)
#
# Thomas Lundqvist, 2026, use freely!

from ollama import chat
import pprint

def get_temperature(city: str) -> str:
  """Get the current temperature for a city
  
  Args:
    city: The name of the city

  Returns:
    The current temperature for the city
  """
  temperatures = {
    "New York": "22°C",
    "London": "15°C",
    "Tokyo": "18°C",
  }
  return temperatures.get(city, "Unknown")

messages = [{"role": "user", "content": "What is the temperature in New York?"}]
print("----------------------- Starting message:")
pprint.pp(messages)

# pass functions directly as tools in the tools list or as a JSON schema
#response = chat(model="gemma3:4b", messages=messages, tools=[get_temperature], think=True)
response = chat(model="qwen3.5", messages=messages, tools=[get_temperature], think=True)
print("----------------------- Model response:")
pprint.pp(response)

messages.append(response.message)
if response.message.tool_calls:
  # only recommended for models which only return a single tool call
  call = response.message.tool_calls[0]
  result = get_temperature(**call.function.arguments)
  # add the tool result to the messages
  messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})
  print("----------------------- Conversation (messages) after tool call:")
  pprint.pp(messages)

  final_response = chat(model="qwen3.5", messages=messages, tools=[get_temperature], think=True)
  print("----------------------- Final response:")
  pprint.pp(final_response)

  print("----------------------- Final answer:")
  print(final_response.message.content)

