import os
import pickle
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Dataset


# ==============================
# 1. Argument parsing
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Collect calibration statistics from Qwen3-VL.")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save calibration data")
    parser.add_argument("--num_samples", type=int, default=256, help="Number of samples to process")
    return parser.parse_args()


# ==============================
# 2. Normalize submodule names
# ==============================
def normalize_name(name):
    if name == "mlp.gate_proj":
        return "mlp.up_proj"
    elif name in ("self_attn.q_proj", "self_attn.v_proj"):
        return "self_attn.k_proj"
    else:
        return name


# ==============================
# 3. Hook class
# ==============================
class SubmoduleHook:
    def __init__(self, layer_idx, submodule_name, save_base_dir):
        self.layer_idx = layer_idx
        self.submodule_name = submodule_name
        self.save_path = os.path.join(save_base_dir, submodule_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.calib = None
        self.total_tokens = 0
        self.hook = None

    def hook_fn(self, module, input, output):
        with torch.no_grad():
            inp = input[0].detach()

            if inp.isnan().any() or inp.isinf().any():
                return

            if inp.dtype != torch.float32:
                inp = inp.float()

            B, S, D = inp.shape
            inp_flat = inp.view(-1, D)
            gram = (inp_flat.t() @ inp_flat).double()

            if self.calib is None:
                self.calib = gram.cpu()
            else:
                self.calib += gram.cpu()

            self.total_tokens += B * S

    def register(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def save_and_close(self):
        if self.calib is not None and self.total_tokens > 0:
            avg_gram = (self.calib / self.total_tokens).float()
            file_path = os.path.join(self.save_path, f"{self.layer_idx}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(avg_gram, f)

        if self.hook:
            self.hook.remove()


# ==============================
# 4. Main pipeline
# ==============================
def main():
    args = parse_args()
    save_base_dir = args.save_dir

    raw_submodule_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]

    normalized_names = sorted(set(normalize_name(n) for n in raw_submodule_names))
    print(f"Will collect activations for: {normalized_names}")

    # ==============================
    # Load dataset
    # ==============================
    print("Loading dataset...")
    dataset = load_dataset("AI4Math/MathVerse", "testmini")["testmini"]
    dataset = Dataset.from_dict(dataset[:args.num_samples])

    # ==============================
    # Load model
    # ==============================
    print("Loading model and processor...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype="bfloat16",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    # ==============================
    # Register hooks
    # ==============================
    hooks = []
    layers = model.model.language_model.layers

    for layer_idx, layer in enumerate(layers):
        for raw_name in raw_submodule_names:
            norm_name = normalize_name(raw_name)

            try:
                submodule = layer.get_submodule(raw_name)
            except AttributeError:
                print(f"Warning: {raw_name} not found in layer {layer_idx}")
                continue

            already_registered = any(
                h.layer_idx == layer_idx and h.submodule_name == norm_name for h in hooks
            )

            if not already_registered:
                hook = SubmoduleHook(layer_idx, norm_name, save_base_dir)
                hook.register(submodule)
                hooks.append(hook)

    print(f"Registered {len(hooks)} hooks.")

    # ==============================
    # Run inference
    # ==============================
    print("Running inference...")
    model.eval()

    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image"]},
                        {"type": "text", "text": sample["question_for_eval"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}],
                },
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                _ = model(**inputs)

        except Exception as e:
            print(f"Skipping sample {i}: {e}")

    # ==============================
    # Save results
    # ==============================
    print("Saving calibration data...")
    for hook in hooks:
        hook.save_and_close()

    print(f"✅ Saved to: {save_base_dir}")


# ==============================
# Entry point
# ==============================
if __name__ == "__main__":
    main()
