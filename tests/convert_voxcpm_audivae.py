import torch
from safetensors.torch import save_file

base_p = 'checkpoints/VoxCPM2'

src = f"{base_p}/audiovae.pth"
dst = f"{base_p}/audiovae.safetensors"

state = torch.load(src, map_location="cpu", weights_only=True)

if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

state = {k: v.contiguous() for k, v in state.items() if torch.is_tensor(v)}

save_file(state, dst)
print(f"saved: {dst}")
