# Getting a REAL Llama-3.2-1B-Instruct running through ExecuTorch (the PS5 hard requirement)

Do this in **WSL2 Ubuntu** (ExecuTorch wheels/build are Linux/macOS). ~2-3 h the first time.
Honest labelling: on your x86 laptop this exercises **XNNPACK**, not Arm KleidiAI. Say so in the report; the
KleidiAI/NEON numbers come from the Arm SBC in Phase 2.

## 1. Weights (gated by Meta's licence)
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                     # after accepting the licence for meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --include "original/*" \
        --local-dir models/llama3.2-1b-instruct
# expect: original/consolidated.00.pth, original/params.json, original/tokenizer.model
```
(Alternative: `pip install llama-stack && llama model download --source meta --model-id Llama3.2-1B-Instruct`.)

## 2. Install ExecuTorch + export
```bash
git clone https://github.com/pytorch/executorch.git && cd executorch
git submodule update --init --recursive
./install_executorch.sh                    # installs the python package + torch
cd ..
python -m executorch.extension.llm.export.export_llm --config scripts/llama32_1b_8da4w.yaml
ls -lh models/llama_et/                    # model.pte  (~1 GB for 8da4w)  -> record the REAL size
cp models/llama3.2-1b-instruct/original/tokenizer.model models/llama_et/tokenizer.model
```
If a YAML key is rejected, fix it per `--help` / the docs for your version — don't guess.

**Fallback (to unblock yourself TODAY, replace later):** a community pre-exported 8da4w XNNPACK `.pte` of this model
exists on Hugging Face (`experimentalmachines/Llama-3.2-1B-Instruct-ExecuTorch`, built for ExecuTorch 1.4.0).
It must match your installed runtime version, and you must keep the Llama licence attribution ("Built with Llama").
Say in the report that the .pte was pre-exported if you use it.

## 3. Build the Llama runner (`llama_main`)
Follow `executorch/examples/models/llama/README.md` in your checkout (cmake presets, e.g.
`cmake --workflow llm-release` then the `llama-release` preset). The binary lands under
`cmake-out/examples/models/llama/llama_main`. Then:
```bash
export LLAMA_RUNNER=$PWD/executorch/cmake-out/examples/models/llama/llama_main
# smoke test — you must see generated text and a "PyTorchObserver {...}" stats line:
$LLAMA_RUNNER --model_path models/llama_et/model.pte --tokenizer_path models/llama_et/tokenizer.model \
   --prompt "<|start_header_id|>user<|end_header_id|>\n\nName one cause of rolled-in scale.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
   --temperature 0 --seq_len 128
```
If your build uses different flag names, set `LLAMA_RUNNER_ARGS` (placeholders: `{model} {tokenizer} {prompt} {seq_len}`).

## 4. Point the pipeline at it
```bash
export SPIKEGUARD_LLAMA_PTE=$PWD/models/llama_et/model.pte
export SPIKEGUARD_LLAMA_TOKENIZER=$PWD/models/llama_et/tokenizer.model
pytest -q && python benchmark_track_a_d.py --images data/test_images --llama-runs 5
```
The dashboard's Stage 2 tile turns from "—" to a measured tok/s as soon as this works.

## Prompt-quality tips for a 1B model
temperature 0 · two-line `ROOT_CAUSE:` / `ACTION:` output (already used) · keep retrieved notes short ·
the disposition is decided by the rule engine and only *explained* by Llama.
