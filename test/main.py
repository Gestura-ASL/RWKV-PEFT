import os

os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

from rwkv.model import RWKV_x070
from rwkv.utils import PIPELINE

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
MODEL_PATH = "/home/karthikssalian/work/RWKV-PEFT/out/rwkv-47-merged"
TOKENIZER_NAME = "rwkv_vocab_v20230424"

DEVICE = "cuda fp16"
CTX_LEN = 512
MAX_GEN_TOKENS = 128
CHUNK_LEN = 256

# ------------------------------------------------------
# LOAD RWKV MODEL
# ------------------------------------------------------


print("Loading model:", MODEL_PATH)
model = RWKV_x070(model=MODEL_PATH, strategy=DEVICE)
pipeline = PIPELINE(model, TOKENIZER_NAME)

print("\nModel loaded!\n")

# ------------------------------------------------------
# STREAMING KEYWORD → SENTENCE GENERATION
# ------------------------------------------------------
def generate_from_keywords(keywords):
    """
    keywords = ["go", "market", "buy", "apple"]
    """
    global model

    # Create prompt
    prompt = "User: " + " ".join(keywords) + "\n\nAssistant:"

    # Prefill
    model_state = None
    tokens = pipeline.encode(prompt)
    tokens = [int(x) for x in tokens]

    # Run prefill
    out = None
    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    # Generation loop
    generated = ""
    last_output_pos = 0
    out_tokens = []

    for i in range(MAX_GEN_TOKENS):
        # Sample next token
        token = pipeline.sample_logits(out)

        # Forward next token
        out, model_state = model.forward([token], model_state)
        out_tokens.append(token)

        # Decode only the last piece
        text = pipeline.decode(out_tokens[last_output_pos:])
        if "\ufffd" not in text:  # Valid UTF-8
            print(text, end="", flush=True)
            generated += text
            last_output_pos = len(out_tokens)

        if text.endswith("\n\n"):
            break

    print()
    return generated

if __name__ == "__main__":
    print("Real-time RWKV keyword sentence generator.")
    print("Type keywords one by one. Type 'run' to generate.")
    print("Example: go, market, buy, apple\n")

    keywords = []

    while True:
        word = input("Keyword list: ").strip()
        print("\nGenerating sentence...\n")
        generate_from_keywords(keywords)
        print("\n")
