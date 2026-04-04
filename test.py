# =========================
# 1. INSTALL (chạy 1 lần)
# =========================
# pip install -q -U torch sentencepiece immutabledict kagglehub
# git clone https://github.com/google/gemma_pytorch.git


# =========================
# 2. IMPORT
# =========================
import os
import sys
import torch
import contextlib
import kagglehub

# =========================
# 3. KAGGLE AUTH
# =========================
os.environ["KAGGLE_USERNAME"] = "blackfox20092006"
os.environ["KAGGLE_KEY"] = "11fefc5d8222ce109207d3d8f689d391"


# =========================
# 4. CONFIG
# =========================
VARIANT = "1b-it"     # 1b-it | 2b-it | 4b-it
MACHINE_TYPE = "cpu"  # "cuda" hoặc "cpu"
CONFIG = VARIANT.split("-")[0]


# =========================
# 5. DOWNLOAD MODEL
# =========================
print("Downloading model...")
weights_dir = kagglehub.model_download(
    f"google/gemma-3/pytorch/gemma-3-{VARIANT}"
)

tokenizer_path = os.path.join(weights_dir, "tokenizer.model")
ckpt_path = os.path.join(weights_dir, "model.ckpt")

assert os.path.isfile(tokenizer_path), "❌ Tokenizer not found"
assert os.path.isfile(ckpt_path), "❌ Checkpoint not found"

print("✅ Model downloaded")


# =========================
# 6. LOAD MODEL
# =========================
sys.path.append("gemma_pytorch/gemma")

from gemma_pytorch.gemma.config import get_model_config
from gemma_pytorch.gemma.gemma3_model import Gemma3ForMultimodalLM

model_config = get_model_config(CONFIG)
model_config.tokenizer = tokenizer_path
model_config.dtype = "float16" if MACHINE_TYPE == "cuda" else "float32"


@contextlib.contextmanager
def set_default_tensor_type(dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


device = torch.device(MACHINE_TYPE)

print("Loading model...")
with set_default_tensor_type(model_config.get_dtype()):
    model = Gemma3ForMultimodalLM(model_config)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model = model.to(device).eval()

print("✅ Model loaded successfully")


# =========================
# 7. CHAT TEMPLATE
# =========================
USER = "<start_of_turn>user\n{prompt}<end_of_turn><eos>\n"
MODEL = "<start_of_turn>model\n"


# =========================
# 8. CHAT LOOP
# =========================
print("\n💬 Gemma Chat (type 'exit' to quit)\n")

history = ""

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        break

    # build prompt
    history += USER.format(prompt=user_input)
    prompt = history + MODEL

    # generate
    output = model.generate(
        prompt,
        device=device,
        output_len=256
    )

    # extract response (clean nhẹ)
    response = output.split("<start_of_turn>model")[-1]
    response = response.replace("<end_of_turn>", "").replace("<eos>", "").strip()

    print(f"Gemma: {response}\n")

    # lưu history
    history += f"<start_of_turn>model\n{response}<end_of_turn><eos>\n"