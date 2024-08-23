import os, sys, re
import datetime
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_community.tools.tavily_search.tool import TavilySearchResults

sys.path.append('../..')
_ = load_dotenv(find_dotenv())


openai.api_key = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

if not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("Please set the environment variable TAVILY_API_KEY")


def get_item_device_wattage(item_name: str, item_model_name: str) -> int:
    """
    Get the device wattage for a given item name and model name
    :param item_name: name of the item
    :param item_model_name: model name of the item
    :return: wattage of the device
    """
    today = datetime.datetime.today().strftime("%D")
    prompt = ChatPromptTemplate(
        [
            ("system", f"You are a helpful assistant. The date today is {today}."),
            ("human", "{user_input}"),
            ("placeholder", "{messages}"),
        ]
    )
    llm_with_tools = llm.bind_tools([tool])
    llm_chain = prompt | llm_with_tools
    @chain
    def tool_chain(user_input: str, config: RunnableConfig):
        input_ = {"user_input": user_input}
        ai_msg = llm_chain.invoke(input_, config=config)
        tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
        return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)
    query_str = f"What is the wattage of a {item_model_name} {item_name}? Output should be as follows: Average Wattage: XX Watts"
    results = tool_chain.invoke(query_str)
    match = re.search(r'Average Wattage:\s*(\d+)\s*Watts', results.content)
    if match:
        wattage = int(match.group(1))
    else:
        wattage = 300
    return wattage