import os
from wcmatch import pathlib
import torch

torch.cuda.empty_cache()
Tensor = torch.Tensor
device = torch.device("cuda" if torch.cuda.is_available() and not bool(os.environ.get("DEEPRKM_CPUONLY", False)) else "cpu")
TensorType = torch.FloatTensor
torch.set_default_tensor_type(TensorType)

ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) # This is your Project Root
OUT_DIR = pathlib.Path("~/out/deeprkm/").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = pathlib.Path("~/data").expanduser()