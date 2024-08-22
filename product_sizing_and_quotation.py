import os, sys
from typing import List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import openai
from tqdm.auto import tqdm
from langchain.schema. document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from Ingestion.ingest import extract_text_and_metadata_from_pdf_document, extract_text_and_metadata_from_csv_document


sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

DB_FAISS_PATH = "vectorstore/db_faiss"
DIR_PATH = "SOLAR_EQUIPMENT_AND_ACCESSORIES"

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")

csv_path = os.path.join(os.path.dirname(__file__),'solar_items.csv')
df = pd.read_csv(csv_path)


def get_unit_price(product_model: str, item_category_code: str) -> float:
    """Fetch the unit price from the dataframe based on product model and item category code."""
    row = df[(df['Product Model'] == product_model) & (df['Item Category Code'] == item_category_code)]
    if not row.empty:
        return row['Unit Price'].values[0]
    else:
        raise ValueError(f"Unit price not found for model {product_model} and category {item_category_code}")   


class Component(BaseModel):
    product_model: str = Field(..., description="Model of the Dayliff product")
    item_category_code: str = Field(..., description="Category code of the item")
    description: str = Field(..., description="Description of the component")
    quantity: int = Field(..., description="Number of units of this component")
    unit_price: float = Field(..., description="Price per unit in KES")
    gross_price: float = Field(..., description="Total cost for the component, calculated as quantity * unit_price")

    @property
    def calculate_gross_price(self) -> float:
        return self.quantity * self.unit_price

    @classmethod
    def from_excel(cls, product_model: str, item_category_code: str, description: str, quantity: int):
        unit_price = get_unit_price(product_model, item_category_code)
        gross_price = quantity * unit_price
        return cls(
            product_model=product_model,
            item_category_code=item_category_code,
            description=description,
            quantity=quantity,
            unit_price=unit_price,
            gross_price=gross_price
        )

class BatteryOption(BaseModel):
    battery_type: str = Field(..., description="Type of battery (e.g., 'Lead Acid' or 'Lithium-Ion')")
    number_of_batteries: int = Field(..., description="Number of batteries in this option")
    battery_capacity: float = Field(..., description="Capacity of each battery in Ah")
    battery_voltage: float = Field(..., description="Voltage of each battery")
    total_capacity: float = Field(..., description="Total capacity of all batteries (number_of_batteries * battery_capacity)")
    components: List[Component] = Field(..., description="List of battery components")

class SolarPanelOption(BaseModel):
    number_of_panels: int = Field(..., description="Number of solar panels")
    panel_wattage: float = Field(..., description="Wattage of each panel")
    total_wattage: float = Field(..., description="Total wattage of all panels (number_of_panels * panel_wattage)")
    components: List[Component] = Field(..., description="List of solar panel components")

class InverterOption(BaseModel):
    inverter_capacity: float = Field(..., description="Capacity of each inverter in kVA")
    number_of_inverters: int = Field(..., description="Number of inverters")
    components: List[Component] = Field(..., description="List of inverter components")

class SolarPowerSolution(BaseModel):
    name: str = Field(..., description="Name of the solar power solution i.e. Solar Power Backup Solution with Lead Acid Batteries or Solar Power Backup Solution with Lithium-Ion Batteries")
    battery: BatteryOption
    solar_panel: SolarPanelOption
    inverter: InverterOption
    other_components: List[Component] = Field(..., description="Other components in the solar power solution")
    subtotal: float = Field(..., description="Subtotal of all components in KES")
    vat: float = Field(..., description="Value Added Tax at 16%")
    grand_total: float = Field(..., description="Total cost including VAT")
    explanation: str = Field(..., description="Detailed explanation of the Solar Power Solution")
    additional_notes: Optional[str] = Field(None, description="Additional notes on warranties, maintenance, etc.")

    @property
    def calculate_subtotal(self) -> float:
        return sum(component.gross_price for component in 
                   self.battery.components + self.solar_panel.components + 
                   self.inverter.components + self.other_components)

    @property
    def calculate_vat(self) -> float:
        return self.calculate_subtotal * 0.16

    @property
    def calculate_grand_total(self) -> float:
        return self.calculate_subtotal + self.calculate_vat

class InverterPowerSolution(BaseModel):
    name: str = Field(..., description="Name of the inverter power solution i.e. Inverter Power Backup Solution")
    inverter: InverterOption
    battery: BatteryOption
    other_components: List[Component] = Field(..., description="Other components in the inverter power solution")
    subtotal: float = Field(..., description="Subtotal of all components in the solution")
    vat: float = Field(..., description="Value Added Tax at 16%")
    grand_total: float = Field(..., description="Total cost including VAT")
    explanation: str = Field(..., description="Detailed explanation of the Inverter Power Solution")
    additional_notes: Optional[str] = Field(None, description="Additional notes on warranties, maintenance, etc.")

    @property
    def calculate_subtotal(self) -> float:
        return sum(component.gross_price for component in 
                   self.inverter.components + self.battery.components + 
                   self.other_components)

    @property
    def calculate_vat(self) -> float:
        return self.calculate_subtotal * 0.16

    @property
    def calculate_grand_total(self) -> float:
        return self.calculate_subtotal + self.calculate_vat

class PowerBackupOptions(BaseModel):
    option1: SolarPowerSolution = Field(..., description="Solar Power Backup Solution with Lead Acid Batteries")
    option2: SolarPowerSolution = Field(..., description="Solar Power Backup Solution with Lithium-Ion Batteries")
    option3: InverterPowerSolution = Field(..., description="Inverter Power Backup Solution")


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
    with open('bm25_retriever.pkl', 'wb') as f:
        pickle.dump(bm25_retriever, f)
    return bm25_retriever


def load_bm25_retriever():
    with open("bm25_retriever.pkl", 'rb') as f:
        bm25_retriever = pickle.load(f)
    return bm25_retriever


def generate_powerbackup_quotation(energy_demand: str):
    """
    Generate a power backup quotation based on the energy demand
    :param energy_demand: energy demand of the client
    :return: power backup quotation
    """
    bm25_retriever = None
    embedding_model = load_embedding_model()
    if not os.path.exists(DB_FAISS_PATH) or not os.path.exists("bm25_retriever.pkl"):
        documents = []
        pdf_files = []
        csv_files = []
        for root, dirs, files in os.walk(DIR_PATH):
            pdf_files.extend([os.path.join(root, f) for f in files if f.endswith('.pdf')])
            csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
            

        for pdf_file in tqdm(pdf_files, desc='Processing PDF files'):
            try:
                df = extract_text_and_metadata_from_pdf_document(pdf_file)
                print(f"Extracted text and metadata from {pdf_file}")
                for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing rows'):
                    file_name = row['Filename']
                    text = row['Text']
                    page_number = row['Page_Number']
                    document = Document(
                        page_content=text,
                        metadata={
                            'id': f"{index}_{file_name}_{page_number}",
                            'type': 'text',
                            'filename': file_name,
                            'page_number': page_number
                        }
                    )
                    documents.append(document)
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")

        for csv_file in tqdm(csv_files, desc='Processing CSV files'):
            try:
                df = extract_text_and_metadata_from_csv_document(csv_file)
                print(f"Extracted text and metadata from {pdf_file}")
                for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing rows'):
                    file_name = row['Filename']
                    text = row['Text']
                    page_number = row['Page_Number']
                    document = Document(
                        page_content=text,
                        metadata={
                            'id': f"{index}_{file_name}_{page_number}",
                            'type': 'text',
                            'filename': file_name,
                            'page_number': page_number
                        }
                    )
                    documents.append(document)
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")

        # Save documents as a pickle file
        with open('documents.pkl', 'wb') as f:
            pickle.dump(documents, f)

        create_vector_db(documents, embedding_model)
        bm25_retriever = initialize_bm25_retriever(documents)
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    faiss_retriever = db.as_retriever()
    if not bm25_retriever:
        bm25_retriever = load_bm25_retriever()
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    output_parser = PydanticOutputParser(pydantic_object=PowerBackupOptions)
    prompt = PromptTemplate(
        template="""
        You are the Davis & Shirtliff Senior Solar Energy Engineer. Your task is to attend to the following request: {question}
        You will consider only Dayliff solar backup products. Use the following pieces of relevant information:\n{context}
        
        Provide a detailed quotation for three power backup solutions, ensuring that each solution includes:
        - A solar power backup solution with lead-acid batteries.
        - A solar power backup solution with lithium-ion batteries.
        - An inverter-based power backup solution.
        
        The quotation should include product models, components, unit prices, quantities, gross prices, subtotal, VAT, and grand total.
        Ensure the output conforms to the specified format.
        {format_instructions}
        """,
        input_variables=["question", "context"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    rag_chain = (
        {"context": ensemble_retriever , "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    query_str = f"Provide a comprehensive analysis and quotation for the three most suitable Dayliff power backup solutions to support an energy demand of {energy_demand} Watt-hours per day in Kenya."
    output = rag_chain.invoke(query_str)
    new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)
    new_output = new_parser.parse(output)
    return new_output
    