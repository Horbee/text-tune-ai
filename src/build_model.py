import os
import sys
import shutil
import subprocess
import argparse

# Get the absolute path of the project root (one level up from src)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.prompts import instruction_v6


def run_command(command, description, cwd=None):
    print(f"\n{'=' * 50}")
    print(f"🚀 STEP: {description}")
    print(f"💻 RUNNING: {' '.join(command)}")
    print(f"{'=' * 50}\n")

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd
    )
    for line in process.stdout:
        print(line, end="")

    process.wait()
    if process.returncode != 0:
        print(f"\n❌ ERROR: Command failed with exit code {process.returncode}")
        exit(1)
    print(f"\n✅ SUCCESS: {description} completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Automate Unsloth LoRA merge, GGUF conversion, and Ollama import."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Name of the adapter folder (e.g., Ministral-3-3B-Instruct-2512-GEC-DPO-v13)",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Final name for the Ollama model (e.g., Text-Tune-Small-v13)",
    )
    args = parser.parse_args()

    ADAPTER_NAME = args.adapter
    FINAL_MODEL_NAME = args.model_name

    # --- Paths Configuration ---
    # Use PROJECT_ROOT to ensure paths are correct regardless of where the script is run from
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    ADAPTER_PATH = os.path.join(MODELS_DIR, ADAPTER_NAME)
    MERGED_SAVE_PATH = os.path.join(MODELS_DIR, "merged_16bit", ADAPTER_NAME)

    BF16_GGUF_PATH = os.path.join(PROJECT_ROOT, f"{ADAPTER_NAME}.BF16.gguf")

    Q4_DIR = os.path.join(ADAPTER_PATH, "gguf-q4_k_m")
    Q4_GGUF_PATH = os.path.join(Q4_DIR, f"{FINAL_MODEL_NAME}.gguf")

    # Executable paths
    LLAMA_CPP_DIR = r"C:\Users\Norbee\.unsloth\llama.cpp"
    CONVERT_SCRIPT = os.path.join(LLAMA_CPP_DIR, "unsloth_convert_hf_to_gguf.py")
    QUANTIZE_EXE = os.path.join(LLAMA_CPP_DIR, r"build\bin\Release\llama-quantize.exe")

    # Ensure output directory for Q4 exists
    os.makedirs(Q4_DIR, exist_ok=True)

    # ==========================================
    # STEP 1: Merge LoRA with Base Model
    # ==========================================
    print(f"\n{'=' * 50}")
    print(f"🚀 STEP: Merging LoRA adapter into 16-bit model")
    print(f"{'=' * 50}\n")

    # Import here to avoid slow startup if just checking help
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=512,
        load_in_4bit=True,
        dtype=None,
        device_map="cpu",
    )

    FastLanguageModel.for_inference(model)
    model.save_pretrained_merged(
        MERGED_SAVE_PATH, tokenizer, save_method="merged_16bit"
    )
    print("\n✅ SUCCESS: Model merged successfully.")

    # ==========================================
    # STEP 2: Convert to BF16 GGUF
    # ==========================================
    convert_cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        MERGED_SAVE_PATH,
        BF16_GGUF_PATH,
    ]
    run_command(convert_cmd, "Convert merged model to BF16 GGUF", cwd=PROJECT_ROOT)

    # ==========================================
    # STEP 3: Quantize to Q4_K_M
    # ==========================================
    quantize_cmd = [QUANTIZE_EXE, BF16_GGUF_PATH, Q4_GGUF_PATH, "Q4_K_M"]
    run_command(quantize_cmd, "Quantize BF16 GGUF to Q4_K_M", cwd=PROJECT_ROOT)

    # ==========================================
    # STEP 4: Create Modelfile & Import to Ollama
    # ==========================================
    print(f"\n{'=' * 50}")
    print(f"🚀 STEP: Generating Modelfile and importing to Ollama")
    print(f"{'=' * 50}\n")

    # Extract system prompt from instruction_v6
    # We split by "TEXT ZUR KORREKTUR:" to separate the system instructions from the user template
    parts = instruction_v6.split("TEXT ZUR KORREKTUR:")
    system_prompt = parts[0].strip()

    # Note: Ollama paths in Modelfile are relative to where `ollama create` is run (PROJECT_ROOT)
    # We need to make Q4_GGUF_PATH relative to PROJECT_ROOT for the Modelfile
    rel_q4_path = os.path.relpath(Q4_GGUF_PATH, PROJECT_ROOT).replace(os.sep, "/")

    modelfile_content = f"""FROM ./{rel_q4_path}

TEMPLATE \"\"\"[INST] {{{{ if .System }}}}{{{{ .System }}}}\n\n{{{{ end }}}}TEXT ZUR KORREKTUR:
{{{{ .Prompt }}}}
Output:[/INST]\"\"\"

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER num_ctx 1024
PARAMETER temperature 0.1
PARAMETER stop "</s>"
PARAMETER stop "[/INST]"
"""

    modelfile_path = os.path.join(PROJECT_ROOT, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    ollama_cmd = ["ollama", "create", FINAL_MODEL_NAME, "-f", "Modelfile"]
    run_command(
        ollama_cmd, f"Importing {FINAL_MODEL_NAME} into Ollama", cwd=PROJECT_ROOT
    )

    # ==========================================
    # STEP 5: Cleanup Intermediate Files
    # ==========================================
    print(f"\n{'=' * 50}")
    print(f"🚀 STEP: Cleaning up intermediate files")
    print(f"{'=' * 50}\n")

    files_to_remove = [BF16_GGUF_PATH, modelfile_path]
    dirs_to_remove = [MERGED_SAVE_PATH]

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ Deleted file: {file}")

    for directory in dirs_to_remove:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"🗑️ Deleted directory: {directory}")

    print("\n🎉 ALL DONE! Your model is ready in Ollama.")
    print(f"Test it with: ollama run {FINAL_MODEL_NAME}")


if __name__ == "__main__":
    main()
