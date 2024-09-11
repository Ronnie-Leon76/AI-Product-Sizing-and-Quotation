import os, sys
import json
import gc
import hashlib
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uvicorn
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain.memory import ConversationBufferMemory
from search_device_wattage import get_item_device_wattage
from product_sizing_and_quotation import generate_powerbackup_quotation, PowerBackupOptions, SolarPowerSolution, InverterPowerSolution, BatteryOption, SolarPanelOption, InverterOption, Component, FilteredPowerBackupOptions

# File paths
CLIENT_COMPLETE_ITEMS_INFORMATION_LIST_JSON_FILE_PATH = "client_complete_items_information_list.json"
CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH = "client_powerbackup_quotation.json"

# Load CSV data
solar_equipment_directory_path = os.path.join(os.path.dirname(__file__), 'SOLAR_EQUIPMENT_AND_ACCESSORIES')

if not os.path.exists(solar_equipment_directory_path):
    raise ValueError(f"The directory {solar_equipment_directory_path} does not exist. Please ensure it's created and contains the necessary files.")


sys.path.append("../..")
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

REDIS_URL = os.getenv("REDIS_URL")

origins = ["*"]

memory = ConversationBufferMemory()

redis_client = redis.StrictRedis.from_url(REDIS_URL)

def get_cache_key(items, location):
    key_str = json.dumps([item.dict() for item in items] + [location])
    return hashlib.md5(key_str.encode()).hexdigest()

def cache_result(key, value, expiry=1800):
    redis_client.setex(key, expiry, json.dumps(value))

def get_cached_result(key):
    result = redis_client.get(key)
    if result:
        return json.loads(result)
    return None

# Pydantic models
class Item(BaseModel):
    item_name: str
    item_model_name: str
    item_quantity: int
    running_hours: float
    location: str

class ItemResponse(BaseModel):
    item_name: str
    item_model: str
    quantity: int
    running_hours: float
    wattage: int
    energy_demand: float

class CompleteItemInformation(BaseModel):
    item_name: str
    item_device_power_rating: int
    item_device_quantity: int
    item_device_total_power: int
    item_device_running_hours: float

# Formatting functions
def format_power_backup_options(quotation: PowerBackupOptions) -> Dict[str, Any]:
    return {
        "option1": format_solar_power_solution(quotation.option1),
        "option2": format_solar_power_solution(quotation.option2),
        "option3": format_inverter_power_solution(quotation.option3)
    }

def format_solar_power_solution(solution: SolarPowerSolution) -> Dict[str, Any]:
    return {
        "name": solution.name,
        "battery": format_battery_option(solution.battery),
        "solar_panel": format_solar_panel_option(solution.solar_panel),
        "inverter": format_inverter_option(solution.inverter),
        "other_components": [format_component(c) for c in solution.other_components],
        "subtotal": solution.subtotal,
        "vat": solution.vat,
        "grand_total": solution.grand_total,
        "explanation": solution.explanation,
        "additional_notes": solution.additional_notes
    }

def format_inverter_power_solution(solution: InverterPowerSolution) -> Dict[str, Any]:
    return {
        "name": solution.name,
        "inverter": format_inverter_option(solution.inverter),
        "battery": format_battery_option(solution.battery),
        "other_components": [format_component(c) for c in solution.other_components],
        "subtotal": solution.subtotal,
        "vat": solution.vat,
        "grand_total": solution.grand_total,
        "explanation": solution.explanation,
        "additional_notes": solution.additional_notes
    }

def format_battery_option(battery: BatteryOption) -> Dict[str, Any]:
    return {
        "battery_type": battery.battery_type,
        "number_of_batteries": battery.number_of_batteries,
        "battery_capacity": battery.battery_capacity,
        "battery_voltage": battery.battery_voltage,
        "total_capacity": battery.total_capacity,
        "components": [format_component(c) for c in battery.components]
    }

def format_solar_panel_option(panel: SolarPanelOption) -> Dict[str, Any]:
    return {
        "number_of_panels": panel.number_of_panels,
        "panel_wattage": panel.panel_wattage,
        "total_wattage": panel.total_wattage,
        "components": [format_component(c) for c in panel.components]
    }

def format_inverter_option(inverter: InverterOption) -> Dict[str, Any]:
    return {
        "inverter_capacity": inverter.inverter_capacity,
        "number_of_inverters": inverter.number_of_inverters,
        "components": [format_component(c) for c in inverter.components]
    }

def format_component(component: Component) -> Dict[str, Any]:
    return {
        "no": component.no,
        "product_model": component.product_model,
        "item_category_code": component.item_category_code,
        "description": component.description,
        "quantity": component.quantity,
        "unit_price": component.unit_price,
        "gross_price": component.gross_price
    }

def format_filtered_power_backup_options(quotation: FilteredPowerBackupOptions) -> Dict[str, Any]:
    formatted_options = {}
    if quotation.option1:
        formatted_options["option1"] = format_solar_power_solution(quotation.option1)
    if quotation.option2:
        formatted_options["option2"] = format_solar_power_solution(quotation.option2)
    if quotation.option3:
        formatted_options["option3"] = format_inverter_power_solution(quotation.option3)
    return formatted_options

# Initialize FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.post("/process-items-and-generate-a-quotation/", response_model=Dict[str, Any])
async def process_items(items: List[Item]) -> Dict[str, Any]:
    """
    Process the items and generate a power backup quotation.
    """
    complete_item_information = []
    total_energy_demand = 0.0
    location = items[0].location
    cache_key = get_cache_key(items, location)

    # Check Redis cache for result
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    # Process each item
    item_responses = []
    for item in items:
        try:
            power_rating = get_item_device_wattage(item.item_name, item.item_model_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        total_power = item.item_quantity * power_rating
        total_energy_demand += total_power * item.running_hours

        item_response = ItemResponse(
            item_name=item.item_name,
            item_model=item.item_model_name,
            quantity=item.item_quantity,
            running_hours=item.running_hours,
            wattage=power_rating,
            energy_demand=total_power * item.running_hours
        )

        item_responses.append(item_response.model_dump())

        complete_item_info = CompleteItemInformation(
            item_name=item.item_name,
            item_device_power_rating=power_rating,
            item_device_quantity=item.item_quantity,
            item_device_total_power=total_power,
            item_device_running_hours=item.running_hours
        )
        complete_item_information.append(complete_item_info)

    # Clear memory after processing items
    gc.collect()

    # Save complete item information to JSON
    try:
        if os.path.exists(CLIENT_COMPLETE_ITEMS_INFORMATION_LIST_JSON_FILE_PATH):
            with open(CLIENT_COMPLETE_ITEMS_INFORMATION_LIST_JSON_FILE_PATH, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load existing item information: {e}")

    # Append new item information to existing data
    updated_data = existing_data + [item.dict() for item in complete_item_information]

    # Save updated item information to JSON
    try:
        with open(CLIENT_COMPLETE_ITEMS_INFORMATION_LIST_JSON_FILE_PATH, "w") as f:
            json.dump(updated_data, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save item information: {e}")
    
    # Generate power backup quotation
    try:
        quotation = generate_powerbackup_quotation(energy_demand=total_energy_demand, conversation_level="First_Quotation", memory=memory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quotation: {e}")

    if not quotation:
        raise HTTPException(status_code=500, detail="Quotation generation returned no result.")

    # Format quotation
    if isinstance(quotation, PowerBackupOptions):
        formatted_quotation = format_power_backup_options(quotation)
    elif isinstance(quotation, dict):
        formatted_quotation = quotation  # Assuming it's already formatted
    else:
        raise HTTPException(status_code=500, detail="Unexpected quotation format.")

    # Add location to the formatted quotation
    formatted_quotation["location"] = location

    # Add item responses to the formatted quotation
    formatted_quotation["items"] = item_responses

    # Load existing quotation data if it exists
    try:
        if os.path.exists(CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH):
            with open(CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH, "r") as f:
                existing_quotations = json.load(f)
            # Ensure existing_quotations is a list
            if not isinstance(existing_quotations, list):
                existing_quotations = [existing_quotations]
        else:
            existing_quotations = []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load existing quotation information: {e}")

    # Append the new quotation to existing data
    existing_quotations.append(formatted_quotation)

    # Save the updated quotations to JSON
    try:
        with open(CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH, "w") as f:
            json.dump(existing_quotations, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save quotation: {e}")

    # Cache the result in Redis
    cache_result(cache_key, formatted_quotation)

    return formatted_quotation


@app.post("/chat-for-quotation-refinement/", response_model=Dict[str, Any])
async def refine_quotation_chat(refinement_query: str) -> Dict[str, Any]:
    """
    Refine the current quotation based on client input.
    """
    cache_key = hashlib.md5(refinement_query.encode()).hexdigest()

    # Check Redis cache for result
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result
    
    # Regenerate the quotation based on the refined input
    try:
        quotation = generate_powerbackup_quotation(conversation_level = "Further_Engagement", memory=memory, customer_request=refinement_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate refined quotation: {e}")
    
    if not quotation:
        raise HTTPException(status_code=500, detail="Quotation generation returned no result.")

    # Format quotation
    if isinstance(quotation, FilteredPowerBackupOptions):
        formatted_quotation = format_filtered_power_backup_options(quotation)
    elif isinstance(quotation, PowerBackupOptions):
        formatted_quotation = format_power_backup_options(quotation)
    elif isinstance(quotation, dict):
        formatted_quotation = quotation  # Assuming it's already formatted
    else:
        raise HTTPException(status_code=500, detail=f"Unexpected quotation format: {type(quotation)}")

    # Save formatted quotation to JSON
    try:
        with open(CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH, "w") as f:
            json.dump(formatted_quotation, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save quotation: {e}")
    
    # Cache the result in Redis
    cache_result(cache_key, formatted_quotation)

    return formatted_quotation

@app.get("/get-complete-items-information-list/", response_model=List[CompleteItemInformation])
async def get_complete_items_information_list() -> List[CompleteItemInformation]:
    """
    Retrieve the complete items information list.
    """
    if not os.path.exists(CLIENT_COMPLETE_ITEMS_INFORMATION_LIST_JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail="Complete items information list not found.")

    try:
        with open(CLIENT_COMPLETE_ITEMS_INFORMATION_LIST_JSON_FILE_PATH, "r") as f:
            data = json.load(f)
        complete_items = [CompleteItemInformation(**item) for item in data]
        return complete_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load item information: {e}")

@app.get("/get-powerbackup-quotation/", response_model=List[Dict[str, Any]])
async def get_powerbackup_quotation() -> List[Dict[str, Any]]:
    """
    Retrieve the latest power backup quotation.
    """
    if not os.path.exists(CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail="Power backup quotation not found.")

    try:
        with open(CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH, "r") as f:
            quotation = json.load(f)
        return quotation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load quotation: {e}")
    
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
