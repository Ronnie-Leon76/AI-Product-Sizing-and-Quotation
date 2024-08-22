import os
import json
import pandas as pd
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from search_device_wattage import get_item_device_wattage
from product_sizing_and_quotation import generate_powerbackup_quotation, PowerBackupOptions, SolarPowerSolution, InverterPowerSolution, BatteryOption, SolarPanelOption, InverterOption, Component

# File paths
CLIENT_COMPLETE_ITEMS_INFORMATION_LIST_JSON_FILE_PATH = "client_complete_items_information_list.json"
CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH = "client_powerbackup_quotation.json"

# Load CSV data
solar_equipment_directory_path = os.path.join(os.path.dirname(__file__), 'SOLAR_EQUIPMENT_AND_ACCESSORIES')

if not os.path.exists(solar_equipment_directory_path):
    raise ValueError(f"The directory {solar_equipment_directory_path} does not exist. Please ensure it's created and contains the necessary files.")

# Pydantic models
class Item(BaseModel):
    item_name: str
    item_model_name: str
    item_quantity: int
    running_hours: float

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
        "product_model": component.product_model,
        "item_category_code": component.item_category_code,
        "description": component.description,
        "quantity": component.quantity,
        "unit_price": component.unit_price,
        "gross_price": component.gross_price
    }

# Initialize FastAPI app
app = FastAPI()

@app.post("/process-items-and-generate-a-quotation/", response_model=Dict[str, Any])
async def process_items(items: List[Item]) -> Dict[str, Any]:
    """
    Process the items and generate a power backup quotation.
    """
    complete_item_information = []
    total_energy_demand = 0.0

    # Process each item
    for item in items:
        try:
            power_rating = get_item_device_wattage(item.item_name, item.item_model_name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        total_power = item.item_quantity * power_rating
        total_energy_demand += total_power * item.running_hours

        complete_item_info = CompleteItemInformation(
            item_name=item.item_name,
            item_device_power_rating=power_rating,
            item_device_quantity=item.item_quantity,
            item_device_total_power=total_power,
            item_device_running_hours=item.running_hours
        )
        complete_item_information.append(complete_item_info)

    # Save complete item information to JSON
    try:
        with open(CLIENT_COMPLETE_ITEMS_INFORMATION_LIST_JSON_FILE_PATH, "w") as f:
            json.dump(
                [item.dict() for item in complete_item_information], 
                f, 
                indent=4
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save item information: {e}")

    # Generate power backup quotation
    try:
        quotation = generate_powerbackup_quotation(total_energy_demand)
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

    # Save formatted quotation to JSON
    try:
        with open(CLIENT_POWERBACKUP_QUOTATION_JSON_FILE_PATH, "w") as f:
            json.dump(formatted_quotation, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save quotation: {e}")

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

@app.get("/get-powerbackup-quotation/", response_model=Dict[str, Any])
async def get_powerbackup_quotation() -> Dict[str, Any]:
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
    uvicorn.run(app, host="0.0.0.0", port=8080)
