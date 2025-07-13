from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import os, sys
import numpy as np
from pydantic import BaseModel

# Import your utils and model code
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import utils.models.models as models
import utils.amc_motion_data_helper as amc_helpers
import utils.numpy_motion_data_helper as np_motion_helpers
import utils.latent_viewer_helper as latent_viewer_helper

app = FastAPI()

# CORS for local dev

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 📢 Mount the static frontend build

@app.get("/ping")
def ping():
    return {"message": "pong"}

# Load model on startup
#base_path = "."
#model_dir = os.path.join(base_path, "GP_models/cmu_data_latent_GP_model_several_motions/")
base_path = os.path.dirname(__file__)

frontend_path = os.path.join( base_path, "../client/dist" )
print("Serving frontend from:", frontend_path)

model_dir = os.path.join(base_path, "models/gplvm/")
with open(os.path.join(model_dir, "data_dict.json"), "r") as json_file:
    loaded_dict = json.load(json_file)

num_frames = loaded_dict["num_frames"]
data_dim = loaded_dict["data_dim"]
latent_dim = loaded_dict["latent_dim"]
num_inducing = loaded_dict["num_inducing"]

model = models.bGPLVM(num_frames, data_dim, latent_dim, num_inducing)
try:
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), weights_only=True))
except Exception as e:
    print("Error loading model:", e)
    raise
model.eval()

# Inducing points and encodings
learned_inducing_points = model.variational_strategy.inducing_points
learned_inducing_points_mean = learned_inducing_points.detach().cpu().numpy()
latent_X = model.X
q_mu = latent_X.q_mu.detach().numpy()
q_log_sigma = latent_X.q_log_sigma.detach().numpy()

# Save q_mu if not exists
# q_mu_path = "../client/public/q_mu.json"
""" 
q_mu_path = os.path.join(os.path.dirname(__file__), "../client/public/q_mu.json")
if not os.path.exists(q_mu_path):
    with open(q_mu_path, "w") as f:
        json.dump(q_mu.tolist(), f)
"""

# Load joints and edges
joints = amc_helpers.parse_asf(base_path + "/01.asf")
motion_amc = amc_helpers.parse_amc(base_path + "/01_01.amc")
_, feature_names = np_motion_helpers.amc_to_numpy_kinematic_tree(motion_amc, joints)
_, joint_names = np_motion_helpers.extract_cartesian_motion_data_from_amc(joints, motion_amc)
edges = np_motion_helpers.extract_edges(joints)

class LatentVector(BaseModel):
    latent: list

@app.post("/generate_skeleton")
def generate_skeleton(data: LatentVector):
    latent_tensor = torch.tensor([data.latent], dtype=torch.float32)
    with torch.no_grad():
        output = model(latent_tensor).mean.cpu().numpy().flatten()
    joint_positions = latent_viewer_helper.output_to_joint_positions(
        output, joints, feature_names, joint_names, data_dim
    )
    return {"joints": joint_positions.tolist(), "edges": edges[0].tolist()}



# 📢 Serve index.html for all other paths (so React router works)
print("Serving frontend from:", frontend_path)
print("Exists?", os.path.exists(frontend_path))

app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")

@app.get("/{full_path:path}")
def serve_react_app(full_path: str):
    index_file = os.path.join(frontend_path, "index.html")
    return FileResponse(index_file)
 

""" 
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
"""