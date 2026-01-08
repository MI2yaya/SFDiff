from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from forecast import generate as generate_hurricane_scenario
from dataloader import load_datapoint
import yaml
from pathlib import Path



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent

MODEL_REGISTRY = {
    "S4_2D": {
        "config": BASE_DIR / "models" / "S4_2D" / "config.yaml"
    },
    "S4_3D": {
        "config": BASE_DIR / "models" / "S4_3D" / "config.yaml"
    },
    "S4_4D": {
        "config": BASE_DIR / "models" / "S4_4D" / "config.yaml"
    }
}


@app.post("/api/generate")
def generate(req: dict):
    try:
        sim_config = req["sim_config"]
        model_id = req["model_id"]
        dataset_id = req["dataset_id"]

        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_id: {model_id}")

        model_config_path = MODEL_REGISTRY[model_id]["config"]
        with open(model_config_path, "r", encoding="utf-8") as fp:
            model_config = yaml.safe_load(fp)


        test_data, generator = load_datapoint(model_config,dataset_id)

        return generate_hurricane_scenario(
            sim_config=sim_config,
            dataset_id=dataset_id,
            model_config=model_config,
            test_data=test_data,
            generator=generator
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))