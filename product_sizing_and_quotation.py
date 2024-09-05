import os, sys
import logging
from typing import List, Optional, Literal
import gc
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import openai
import requests
from requests.auth import HTTPBasicAuth
from tqdm.auto import tqdm
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from Ingestion.ingest import (
    extract_text_and_metadata_from_pdf_document,
    extract_text_and_metadata_from_csv_document,
)


sys.path.append("../..")
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]
API_USERNAME = os.getenv('API_USERNAME')
API_PASSWORD = os.getenv('API_PASSWORD')
BASE_URL = os.getenv('BASE_URL')

DB_FAISS_PATH = "vectorstore/db_faiss"
DIR_PATH = "SOLAR_EQUIPMENT_AND_ACCESSORIES"

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")

csv_path = os.path.join(os.path.dirname(__file__), "solar_items.csv")
df = pd.read_csv(csv_path)

unique_no_values = df["No."].unique().tolist()
unique_product_models = df["Product Model"].unique().tolist()
unique_item_category_codes = df["Item Category Code"].unique().tolist()

NoOptions = Literal[tuple(unique_no_values)]
ProductModelsOptions = Literal[tuple(unique_product_models)]
ItemCategoryCodesOptions = Literal[tuple(unique_item_category_codes)]

def load_item_descriptions():
    csv_path = os.path.join(os.path.dirname(__file__), "solar_items.csv")
    df = pd.read_csv(csv_path)
    return df[["No.", "Description"]]

def format_item_descriptions(df):
    return "\n".join([f"{row['No.']} - {row['Description']}" for _, row in df.iterrows()])

def get_unit_price(no: str) -> tuple:
    """Fetch the unit price from the dataframe based on product model and item category code."""
    item_details = fetch_item_details(no)
    if item_details and "unit_price" in item_details:
        return item_details["unit_price"], item_details["inventory"], item_details["description"], item_details["item_category_code"], item_details["product_model"]
    else:
        product_model = item_details.get("product_model", "")
        item_category_code = item_details.get("item_category_code", "")
        if product_model and item_category_code:
            row = df[
                (df["Product Model"] == product_model)
                & (df["Item Category Code"] == item_category_code)
            ]
            if not row.empty:
                return row["Unit Price"].values[0], item_details["inventory"], item_details["description"], item_details["item_category_code"], item_details["product_model"]
            else:
                raise ValueError(
                    f"Unit price not found for model {product_model} and category {item_category_code}"
                )

def fetch_item_details(no: str, username: str = API_USERNAME, password: str = API_PASSWORD) -> dict:
    """Fetch item details from the API based on the 'no' field, using basic authentication."""
    base_url = BASE_URL
    params = {"$filter": f"No eq '{no}'"}
    
    try:
        response = requests.get(
            base_url, 
            params=params, 
            auth=HTTPBasicAuth(username, password)
        )
        response.raise_for_status()
        data = response.json()
        if 'value' in data and len(data['value']) > 0:
            item_data = data['value'][0]
            return {
                'no': item_data.get('No', ''),
                'inventory': int(item_data.get('Inventory', 0)),
                'unit_price': float(item_data.get('Unit_Price', 0)),
                'description': item_data.get('Description', ''),
                'item_category_code': item_data.get('Item_Category_Code', ''),
                'product_model': item_data.get('Product_Model', '')
            }
        else:
            return {}
    except requests.RequestException as e:
        print(f"Error fetching data for item {no}: {str(e)}")
        return {}

    
def validate_quantity(requested_quantity: int, available_inventory: int) -> int:
    """Ensure the requested quantity does not exceed the available inventory."""
    if requested_quantity > available_inventory:
        print(f"Warning: Requested quantity ({requested_quantity}) exceeds available inventory ({available_inventory}). Adjusting quantity to available inventory.")
        return available_inventory
    return requested_quantity

class Component(BaseModel):
    no: NoOptions = Field(..., description="Product number of the component. Must be one of the predefined options.")
    product_model: ProductModelsOptions = Field(..., description="Model of the Dayliff product")
    item_category_code: ItemCategoryCodesOptions = Field(..., description="Category code of the item")
    description: str = Field(..., description="Description of the component")
    quantity: int = Field(..., description="Number of units of this component")
    unit_price: float = Field(0.0, description="Price per unit in KES")
    gross_price: float = Field(0.0, description="Total cost for the component")


class BatteryOption(BaseModel):
    battery_type: str = Field(
        ..., description="Type of battery (e.g., 'Lead Acid' or 'Lithium-Ion')"
    )
    number_of_batteries: int = Field(
        ..., description="Number of batteries in this option"
    )
    battery_capacity: float = Field(..., description="Capacity of each battery in Ah")
    battery_voltage: float = Field(..., description="Voltage of each battery")
    total_capacity: float = Field(
        ...,
        description="Total capacity of all batteries (number_of_batteries * battery_capacity)",
    )
    components: List[Component] = Field(..., description="List of battery components")


class SolarPanelOption(BaseModel):
    number_of_panels: int = Field(..., description="Number of solar panels")
    panel_wattage: float = Field(..., description="Wattage of each panel")
    total_wattage: float = Field(
        ...,
        description="Total wattage of all panels (number_of_panels * panel_wattage)",
    )
    components: List[Component] = Field(
        ..., description="List of solar panel components"
    )


class InverterOption(BaseModel):
    inverter_capacity: float = Field(
        ..., description="Capacity of each inverter in kVA"
    )
    number_of_inverters: int = Field(..., description="Number of inverters")
    components: List[Component] = Field(..., description="List of inverter components")


class SolarPowerSolution(BaseModel):
    name: str = Field(
        ...,
        description="Name of the solar power solution i.e. Solar Power Backup Solution with Lead Acid Batteries or Solar Power Backup Solution with Lithium-Ion Batteries",
    )
    battery: BatteryOption
    solar_panel: SolarPanelOption
    inverter: InverterOption
    other_components: List[Component] = Field(
        ..., description="Other components in the solar power solution"
    )
    subtotal: float = Field(0.0, description="Subtotal of all components in KES")
    vat: float = Field(0.0, description="Value Added Tax at 16%")
    grand_total: float = Field(0.0, description="Total cost including VAT")
    explanation: str = Field(
        ..., description="Detailed explanation of the Solar Power Solution"
    )
    additional_notes: Optional[str] = Field(
        None, description="Additional notes on warranties, maintenance, etc."
    )


class InverterPowerSolution(BaseModel):
    name: str = Field(
        ...,
        description="Name of the inverter power solution i.e. Inverter Power Backup Solution",
    )
    inverter: InverterOption
    battery: BatteryOption
    other_components: List[Component] = Field(
        ..., description="Other components in the inverter power solution"
    )
    subtotal: float = Field(
        0.0, description="Subtotal of all components in the solution"
    )
    vat: float = Field(0.0, description="Value Added Tax at 16%")
    grand_total: float = Field(0.0, description="Total cost including VAT")
    explanation: str = Field(
        ..., description="Detailed explanation of the Inverter Power Solution"
    )
    additional_notes: Optional[str] = Field(
        None, description="Additional notes on warranties, maintenance, etc."
    )


class PowerBackupOptions(BaseModel):
    option1: SolarPowerSolution = Field(
        ..., description="Solar Power Backup Solution with Lead Acid Batteries"
    )
    option2: SolarPowerSolution = Field(
        ..., description="Solar Power Backup Solution with Lithium-Ion Batteries"
    )
    option3: InverterPowerSolution = Field(
        ..., description="Inverter Power Backup Solution"
    )


if not os.path.exists(DIR_PATH):
    print(f"Documents Directory path {DIR_PATH} does not exist")
    sys.exit(1)


def load_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-large")


def create_vector_db(documents, embedding_model):
    # Create a vector store
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(DB_FAISS_PATH)


def initialize_bm25_retriever(documents):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    # Save bm25_retriever as a pickle file
    with open("bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    return bm25_retriever


def load_bm25_retriever():
    with open("bm25_retriever.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
    return bm25_retriever


def generate_powerbackup_quotation(
    energy_demand=None,
    conversation_level: str = None,
    memory: ConversationBufferMemory = None,
    customer_request=None,
    location=None,
):
    """
    Generate a power backup quotation based on the energy demand
    :param energy_demand: energy demand of the client
    :return: power backup quotation
    """
    try:
        bm25_retriever = None
        embedding_model = load_embedding_model()
        item_descriptions_df = load_item_descriptions()
        item_descriptions_string = format_item_descriptions(item_descriptions_df)
        if not os.path.exists(DB_FAISS_PATH) or not os.path.exists("bm25_retriever.pkl"):
            documents = []
            pdf_files = []
            csv_files = []
            for root, dirs, files in os.walk(DIR_PATH):
                pdf_files.extend(
                    [os.path.join(root, f) for f in files if f.endswith(".pdf")]
                )
                csv_files.extend(
                    [os.path.join(root, f) for f in files if f.endswith(".csv")]
                )

            for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
                try:
                    df = extract_text_and_metadata_from_pdf_document(pdf_file)
                    print(f"Extracted text and metadata from {pdf_file}")
                    for index, row in tqdm(
                        df.iterrows(), total=len(df), desc="Processing rows"
                    ):
                        file_name = row["Filename"]
                        text = row["Text"]
                        page_number = row["Page_Number"]
                        document = Document(
                            page_content=text,
                            metadata={
                                "id": f"{index}_{file_name}_{page_number}",
                                "type": "text",
                                "filename": file_name,
                                "page_number": page_number,
                            },
                        )
                        documents.append(document)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")

            for csv_file in tqdm(csv_files, desc="Processing CSV files"):
                try:
                    df = extract_text_and_metadata_from_csv_document(csv_file)
                    print(f"Extracted text and metadata from {pdf_file}")
                    for index, row in tqdm(
                        df.iterrows(), total=len(df), desc="Processing rows"
                    ):
                        file_name = row["Filename"]
                        text = row["Text"]
                        page_number = row["Page_Number"]
                        document = Document(
                            page_content=text,
                            metadata={
                                "id": f"{index}_{file_name}_{page_number}",
                                "type": "text",
                                "filename": file_name,
                                "page_number": page_number,
                            },
                        )
                        documents.append(document)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")

            # Save documents as a pickle file
            with open("documents.pkl", "wb") as f:
                pickle.dump(documents, f)

            create_vector_db(documents, embedding_model)
            bm25_retriever = initialize_bm25_retriever(documents)
        db = FAISS.load_local(
            DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
        )
        faiss_retriever = db.as_retriever()
        if not bm25_retriever:
            bm25_retriever = load_bm25_retriever()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        output_parser = PydanticOutputParser(pydantic_object=PowerBackupOptions)
        memory = ConversationBufferMemory()
        # Set up the prompt with location included if provided
        prompt_template = """
            You are the Davis & Shirtliff Senior Solar Energy Engineer. Your task is to attend to the following request: {question}
            You will consider only Dayliff solar backup products. Use the following pieces of relevant information:\n{context}
            
            Chat History:
            {chat_history}

            Here is a list of all available Dayliff components No. with their corresponding descriptions:
            {item_descriptions}

            Provide a detailed quotation for three power backup solutions{location_clause}, ensuring that each solution includes:
            - A solar power backup solution with lead-acid batteries.
            - A solar power backup solution with lithium-ion batteries.
            - An inverter-based power backup solution.

            For each component in the solutions, provide the following information:
            - Accurate product number (no)
            - Correct product model
            - Appropriate item category code
            - Detailed description (ensure it matches exactly with the provided item descriptions)
            - Required quantity

            When specifying components, always use the exact No. and Description pairs from the list provided above.
            Do not include unit prices, gross prices, or VAT calculations in your response. These will be handled separately.
            Ensure the output conforms to the specified format.
            {format_instructions}
        """

        location_clause = f" to be installed in {location}" if location else ""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context", "chat_history", "item_descriptions"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions(),
                "location_clause": location_clause,
            },
        )

        rag_chain = (
            {
                "context": ensemble_retriever,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: memory.chat_memory.messages,
                "item_descriptions": lambda x: item_descriptions_string,
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        if conversation_level == "First_Quotation":
            query_str = f"Provide a comprehensive analysis and quotation for the three most suitable Dayliff power backup solutions to support an energy demand of {energy_demand} Watt-hours per day{location_clause}."
            logger.debug(f"Query string: {query_str}")
            output = rag_chain.invoke(query_str)
            logger.debug(f"RAG chain output: {output}")
            memory.save_context({"input": query_str}, {"output": output})
            new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)
            quotation = new_parser.parse(output)
            logger.debug(f"Parsed quotation: {quotation}")
            if quotation is None:
                raise ValueError("Quotation parsing resulted in None")
            final_quotation = add_pricing_information(quotation)
            logger.debug(f"Final quotation with pricing: {final_quotation}")
            return final_quotation
        elif conversation_level == "Further_Engagement":
            query_str = customer_request
            output = rag_chain.invoke(query_str)
            memory.save_context({"input": query_str}, {"output": output})
            new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)
            quotation = new_parser.parse(output)
            return add_pricing_information(quotation)
    except Exception as e:
        logger.exception(f"Error in generate_powerbackup_quotation: {str(e)}")
        raise

def calculate_subtotal(solution):
    """Calculate the subtotal for a power solution, accounting for different types of solutions."""
    subtotal = 0.0
    
    if isinstance(solution, InverterPowerSolution):
        component_lists = [solution.inverter.components, solution.battery.components, solution.other_components]
    else:
        component_lists = [solution.solar_panel.components, solution.battery.components, solution.inverter.components, solution.other_components]
    
    for component_list in component_lists:
        for component in component_list:
            unit_price, inventory, description, item_category_code, product_model = get_unit_price(component.no)
            if unit_price > 0:
                component.unit_price = unit_price
                component.description = description
                component.item_category_code = item_category_code
                component.product_model = product_model
                quantity = component.quantity
                valid_quantity = validate_quantity(quantity, inventory)
                component.quantity = valid_quantity
                component.gross_price = unit_price * component.quantity
                subtotal += component.gross_price
            else:
                print(f"Warning: Unable to calculate price for component {component.no}")
    
    return subtotal


def add_pricing_information(quotation: PowerBackupOptions):
    """Add pricing details like subtotal, VAT, and grand total for the solutions."""
    for option in [quotation.option1, quotation.option2, quotation.option3]:
        option.subtotal = calculate_subtotal(option)
        option.vat = option.subtotal * 0.16
        option.grand_total = option.subtotal + option.vat
    return quotation
