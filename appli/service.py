from fastapi import FastAPI
from typing_extensions import Annotated
from pydantic import Field
from typing import Literal
import pandas as pd
import bentoml

# Charger le modèle
model_ref = bentoml.sklearn.get("energy_consumption_model:latest")
model = model_ref.load_model()
FEATURES = model_ref.custom_objects["features"]

app = FastAPI(title="Building Energy Prediction API")

@app.post("/predict")
def predict(
    PrimaryPropertyType: Literal[
    "Medical Office",
    "Self-Storage Facility",
    "Small- and Mid-Sized Office",
    "Large Office",
    "Warehouse",
    "Other",
    "Distribution Center",
    "K-12 School",
    "Senior Care Community",
    "Mixed Use Property",
    "Hotel",
    "University",
    "Retail Store",
    "Restaurant",
    "Supermarket / Grocery Store",
    "Worship Facility",
    "Residence Hall",
    "Hospital",
    "Laboratory",
    "Refrigerated Warehouse",
],
    Neighborhood: Literal[
    "DELRIDGE",
    "GREATER DUWAMISH",
    "DOWNTOWN",
    "SOUTHWEST",
    "MAGNOLIA / QUEEN ANNE",
    "NORTH",
    "SOUTHEAST",
    "LAKE UNION",
    "EAST",
    "CENTRAL",
    "NORTHWEST",
    "NORTHEAST",
    "Delridge",
    "BALLARD",
    "Central",
],

    NumberofBuildings: Annotated[int, Field(ge=1, description=">= 1")],
    NumberofFloors: Annotated[int, Field(ge=1)],

    PropertyGFABuilding_s: Annotated[float, Field(gt=0)],

    UsesSteam: Annotated[int, Field(ge=0, le=1)],
    UsesElectricity: Annotated[int, Field(ge=0, le=1)],
    UsesNaturalGas: Annotated[int, Field(ge=0, le=1)],

    BuildingAge: Annotated[int, Field(ge=0, le=300)]
):
    data = {
        "PrimaryPropertyType": PrimaryPropertyType,
        "Neighborhood": Neighborhood,
        "NumberofBuildings": NumberofBuildings,
        "NumberofFloors": NumberofFloors,
        "PropertyGFABuilding(s)": PropertyGFABuilding_s,
        "UsesSteam": UsesSteam,
        "UsesElectricity": UsesElectricity,
        "UsesNaturalGas": UsesNaturalGas,
        "BuildingAge": BuildingAge,
    }

    df = pd.DataFrame([data])
    df = df[FEATURES]
    prediction = model.predict(df)[0]

    return {"SiteEnergyUseWN(kBtu)": float(prediction)}
