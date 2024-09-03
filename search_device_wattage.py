import os
import re
import json
import datetime
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

if not serper_api_key:
    raise ValueError("Please set the environment variable SERPER_API_KEY")

# Constants
POWER_RATING_JSON_FILE_PATH = "device_power_rating.json"
DEFAULT_WATTAGE = 200  # New constant for default wattage

# Initialize models and tools
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="Use this tool for search-based queries."
    )
]
agent = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH)

def get_item_device_wattage(item_name: str, item_model_name: str) -> int:
    """
    Get the device wattage for a given item name and model name.
    :param item_name: name of the item
    :param item_model_name: model name of the item
    :return: wattage of the device
    """
    # First, try to get wattage from JSON file
    if os.path.exists(POWER_RATING_JSON_FILE_PATH):
        try:
            with open(POWER_RATING_JSON_FILE_PATH, "r") as f:
                power_rating_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            power_rating_data = {}
    else:
        power_rating_data = {}
    
    wattage = power_rating_data.get(item_name, {}).get("average_wattage")
    
    if wattage is not None:
        return wattage
    
    # If wattage is not found in JSON, use Serper search
    query_str = f"What is the average wattage of a {item_name} {item_model_name}? Output should be as follows: Average Wattage: XX Watts"
    try:
        results = agent.run(query_str, handle_parsing_errors=True)
    except Exception as e:
        print(f"Error during agent.run: {e}")
        return DEFAULT_WATTAGE
    
    # Extract wattage from search results
    wattage = extract_wattage(results)
    
    if wattage is None:
        fallback_query_str = (
            f"Please clarify the wattage for {item_model_name} {item_name}, "
            "or provide an estimated wattage."
        )
        try:
            fallback_results = agent.run(fallback_query_str, handle_parsing_errors=True)
            wattage = extract_wattage(fallback_results)
        except Exception as e:
            print(f"Error during fallback agent.run: {e}")
            return DEFAULT_WATTAGE
    
    if wattage is None:
        wattage = DEFAULT_WATTAGE  # Use the default wattage constant
    
    # Update JSON with new device data
    if item_name not in power_rating_data:
        power_rating_data[item_name] = {"average_wattage": wattage}
        try:
            with open(POWER_RATING_JSON_FILE_PATH, "w") as f:
                json.dump(power_rating_data, f, indent=4)
        except IOError as e:
            print(f"Error writing JSON file: {e}")
    
    return wattage

def extract_wattage(text: str) -> int:
    match = re.search(r"Average Wattage:\s*(\d+)\s*Watts", text)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", text)
    if numbers:
        wattage_values = [int(num) for num in numbers]
        return sum(wattage_values) // len(wattage_values)  # Calculate average
    return None