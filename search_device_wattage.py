import os
import re
import json
import datetime
import requests
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.schema.output_parser import OutputParserException
from langchain.tools import StructuredTool

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

if not serper_api_key:
    raise ValueError("Please set the environment variable SERPER_API_KEY")

# Constants
POWER_RATING_JSON_FILE_PATH = "device_power_rating.json"
DEFAULT_WATTAGE = 200  # New constant for default wattage

class WattageResponse(BaseModel):
    """
    Final response to the question being asked
    """
    wattage: int = Field(
        ...,
        title="Wattage",
        description="The wattage of the device in Watts."
    )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant tasked with finding accurate information about electronic devices and their power consumption. Use the Google Search tool when you need to look up specific details."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# Initialize models and tools
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")
search = GoogleSerperAPIWrapper()


def fallback_search(query: str) -> str:
    """Perform a fallback search using a simple web request."""
    try:
        response = requests.get(f"https://www.google.com/search?q={query}")
        return response.text
    except Exception as e:
        return f"Error performing fallback search: {str(e)}"

def google_search(query: str, tags: list = None, max_concurrency: int = 1) -> str:
    """Perform a Google search with fallback."""
    try:
        result = search.run(query)
        if not result:
            print("No results from GoogleSerperAPIWrapper, using fallback search.")
            return fallback_search(query)
        return result
    except Exception as e:
        print(f"Error with GoogleSerperAPIWrapper: {str(e)}. Using fallback search.")
        return fallback_search(query)

tools = [
    StructuredTool.from_function(
        func=google_search,
        name="Google_Search",
        description="Useful for when you need to answer questions about current events or the current state of the world. Use this to search for specific information online."
    )
]


wattage_response_tool = convert_to_openai_function(WattageResponse)
llm_with_tools = llm.bind_functions(tools + [wattage_response_tool])
# agent = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)


def parse(output):
    try:
        if "function_call" not in output.additional_kwargs:
            return AgentFinish(return_values={"output": output.content}, log=output.content)

        function_call = output.additional_kwargs["function_call"]
        name = function_call["name"]
        inputs = json.loads(function_call["arguments"])

        if name == "WattageResponse":
            return AgentFinish(return_values=inputs, log=str(function_call))
        else:
            return AgentActionMessageLog(
                tool=name, tool_input=inputs, log="", message_log=[output]
            )
    except Exception as e:
        raise OutputParserException(f"Error parsing output: {str(e)}")

agent = (
    {
        "input": lambda x: x["input"],
        # Format agent scratchpad from intermediate steps
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | parse
)

agent_executor = AgentExecutor(tools=tools, agent=agent, max_iterations=4)

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
    
    # Concatenate the item name and model name
    item_name_ = f"{item_model_name} {item_name}"
    wattage = power_rating_data.get(item_name_, {}).get("average_wattage")
    
    if wattage is not None:
        return wattage
    
    query_str = f"What is the average power consumption in watts for a {item_model_name} {item_name}?"
    try:
        results = agent_executor.invoke({"input": query_str})
        wattage = results.get("wattage")
        if wattage is None:
            wattage = extract_wattage(results.get("output", ""))
    except Exception as e:
        print(f"Error during agent_executor.invoke: {e}")
        return DEFAULT_WATTAGE
    
    # Extract wattage from search results
    # wattage = extract_wattage(results)
    
    # if wattage is None:
    #     fallback_query_str = (
    #         f"Please clarify the wattage for {item_model_name} {item_name}, "
    #         "or provide an estimated wattage."
    #     )
    #     try:
    #         fallback_results = agent.run(fallback_query_str)
    #         wattage = extract_wattage(fallback_results)
    #     except Exception as e:
    #         print(f"Error during fallback agent.run: {e}")
    #         return DEFAULT_WATTAGE
    
    if wattage is None:
        wattage = DEFAULT_WATTAGE  # Use the default wattage constant
    
    # Update JSON with new device data
    if item_name_ not in power_rating_data:
        power_rating_data[item_name_] = {"average_wattage": wattage}
        try:
            with open(POWER_RATING_JSON_FILE_PATH, "w") as f:
                json.dump(power_rating_data, f, indent=4)
        except IOError as e:
            print(f"Error writing JSON file: {e}")
    
    return wattage

def extract_wattage(text: str) -> int:
    match = re.search(r"(\d+)\s*(?:W|Watts?|watt)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*W", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", text)
    if numbers:
        wattage_values = [int(num) for num in numbers if int(num) < 10000]
        if wattage_values:
            return sum(wattage_values) // len(wattage_values)
    
    return None