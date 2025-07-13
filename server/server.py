from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import os
import sys
import numpy as np
from pydantic import BaseModel

# === Import your utils and model code ===
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import utils.models.models as models
import utils.amc_motion_data_helper as amc_helpers
import utils.numpy_motion_data_helper as np_motion_helpers
import utils.latent_viewer_helper as latent_viewer_helper

app = FastAPI()

# === CORS for local dev ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"message": "pong"}

# === Paths ===
base_path = os.path.dirname(__file__)
frontend_path = os.path.join(base_path, "../client/dist")
print("Serving frontend from:", frontend_path)

# === Load models ===
model_registry = {}

# --- GPLVM Model ---
gplvm_dir = os.path.join(base_path, "models/gplvm/")
with open(os.path.join(gplvm_dir, "data_dict.json"), "r") as json_file:
    loaded_dict = json.load(json_file)

num_frames = loaded_dict["num_frames"]
data_dim = loaded_dict["data_dim"]
latent_dim = loaded_dict["latent_dim"]
num_inducing = loaded_dict["num_inducing"]

gplvm_model = models.bGPLVM(num_frames, data_dim, latent_dim, num_inducing)
gplvm_model.load_state_dict(
    torch.load(os.path.join(gplvm_dir, "model.pth"), weights_only=True)
)
gplvm_model.eval()

# Load motion data
joints = amc_helpers.parse_asf(base_path + "/01.asf")
motion_amc = amc_helpers.parse_amc(base_path + "/01_01.amc")
_, feature_names = np_motion_helpers.amc_to_numpy_kinematic_tree(motion_amc, joints)
_, joint_names = np_motion_helpers.extract_cartesian_motion_data_from_amc(joints, motion_amc)
edges = np_motion_helpers.extract_edges(joints)

model_registry["gplvm"] = {
    "type": "skeleton",
    "model": gplvm_model,
    "joints": joints,
    "edges": edges,
    "data_dim": data_dim,
    "feature_names": feature_names,
    "joint_names": joint_names,
}

# --- Wasserstein VAE Model ---
wvae_dir = os.path.join(base_path, "models/wasserstein_vae/")
wasserstein_vae_model = models.ConvVAE()
checkpoint = torch.load(os.path.join(wvae_dir, "conv_vae_sinkhorn.pth"))
wasserstein_vae_model.load_state_dict(checkpoint["model_state_dict"])
wasserstein_vae_model.eval()

model_registry["wasserstein_vae"] = {
    "type": "image",
    "model": wasserstein_vae_model,
}

# --- Additional VAE Model ---
vae_dir = os.path.join(base_path, "models/vae/")
vae_model = models.ConvVAE()
checkpoint = torch.load(os.path.join(vae_dir, "conv_vae_mse.pth"))
vae_model.load_state_dict(checkpoint["model_state_dict"])
vae_model.eval()

model_registry["vae"] = {
    "type": "image",
    "model": vae_model,
}

# === API Schema ===
class LatentVector(BaseModel):
    latent: list

# === Generic Generate Endpoint ===
@app.post("/api/models/{model_id}/generate")
def generate_output(model_id: str, data: LatentVector):
    if model_id not in model_registry:
        return {"error": f"Model '{model_id}' not found"}

    entry = model_registry[model_id]
    latent_tensor = torch.tensor([data.latent], dtype=torch.float32)

    if entry["type"] == "skeleton":
        with torch.no_grad():
            output = entry["model"](latent_tensor)
        out_numpy = output.mean.cpu().numpy().flatten()
        joint_positions = latent_viewer_helper.output_to_joint_positions(
            out_numpy,
            entry["joints"],
            entry["feature_names"],
            entry["joint_names"],
            entry["data_dim"],
        )
        return {
            "joints": joint_positions.tolist(),
            "edges": entry["edges"][0].tolist(),
        }

    elif entry["type"] == "image":
        with torch.no_grad():
            recon = entry["model"].decode(latent_tensor)

        recon = recon.squeeze().cpu().numpy()

        # Convert to PNG base64
        import io, base64
        from PIL import Image

        img = Image.fromarray((recon * 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        b64_string = base64.b64encode(img_bytes).decode("utf-8")
        return {"image": b64_string}

    else:
        return {"error": f"Unknown model type '{entry['type']}'"}
        
# === Serve Frontend ===
print("Serving frontend from:", frontend_path)
print("Exists?", os.path.exists(frontend_path))
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")

@app.get("/{full_path:path}")
def serve_react_app(full_path: str):
    index_file = os.path.join(frontend_path, "index.html")
    return FileResponse(index_file)
