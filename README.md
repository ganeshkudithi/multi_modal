# multi_modal

#kishord account paramrudra.iitb

directory -- ```/scratch/kishord/multimodal/MatCLIP```

env activation -- ```source matclip_env/bin/activate```

command to run model imange generation and validation -- ```python3 matclip_generate_and_validate.py         --params "WAAM, pearlitic steel, current 118A, travel speed 350mm/min"         --reference vlasov_fig6a_ref.png         --label "WAAM PearlSteel Vlasov2026"         --seed 200```

the observation --


1. The initial model from the directory is the waam_hybrid model and the diffusion model and now we have downloaded the mistrial and diffusion model to our directory and using that path

2. Once the model weights are loaded, it ia taking lot of time to  at the step ```[RAG] Load failed: [Errno 2] No such file or directory: '/scratch/kishord/multimodal/MatCLIP/rag_store/bm25_index.pkl' — will skip RAG context```.

The overall log --
```
(matclip_env) [kishord@login06 MatCLIP]$ python3 matclip_generate_and_validate.py         --params "WAAM, pear
litic steel, current 118A, travel speed 350mm/min"         --reference vlasov_fig6a_ref.png         --label "W
AAM PearlSteel Vlasov2026"         --seed 200

============================================================
  MatCLIP Generate + Validate
  Label      : WAAM PearlSteel Vlasov2026
  Run ID     : waam_pearlsteel_vlasov2026_20260623_113435
  Params     : WAAM, pearlitic steel, current 118A, travel speed 350mm/min
  Morphology : AUTO (LLM will predict)
  Skip LLM   : False
  Scale      : 20um
  Reference  : vlasov_fig6a_ref.png
  Invert ref : False
============================================================

============================================================
  STEP 1: Generating microstructure (10µm + 20µm + 50µm)
============================================================
  CMD: /scratch/kishord/llm/MatCLIP/matclip_env/bin/python3 /scratch/kishord/multimodal/MatCLIP/infer.py WAAM, pearlitic steel, current 118A, travel speed 350mm/min --id waam_pearlsteel_vlasov2026_20260623_113435 --multiscale --seed 200


============================================================
  MatCLIP Inference
  Run ID  : waam_pearlsteel_vlasov2026_20260623_113435
  Device  : cpu
  Process : WAAM
============================================================

  INPUT PARAMS:
  WAAM, pearlitic steel, current 118A, travel speed 350mm/min

── STEP 1: Morphology ───────────────────────────────────
[LLM] Loading waam_gemma2...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████| 291/291 [00:02<00:00, 103.37it/s]
[LLM] Ready.
[RAG] Load failed: [Errno 2] No such file or directory: '/scratch/kishord/multimodal/MatCLIP/rag_store/bm25_index.pkl' — will skip RAG context
  [RAG] Context used: False | LLM morphology: CELLULAR
  [LOCK] CELLULAR -> ACICULAR (rule: WAAM+pearlitic_steel)
  Morphology  : ACICULAR
  Visual desc : The microstructure exhibits a columnar grain structure with well-defined boundaries, indicative of d...
  Properties  : To provide an answer, I have researched published literature on WAAM (Wire Arc Additive Manufacturin...
  Time: 457.5s

── STEP 2: MatCLIP Retrieval ────────────────────────────
[MatCLIP] Loading model...
/scratch/kishord/multimodal/MatCLIP/matclip_pipeline_v5.py:598: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state_dict = torch.load(MATCLIP_CKPT, map_location=device)
[MatCLIP] Loading FAISS index...
[MatCLIP] Ready. 10760 vectors indexed.
  [1] sim=0.2443 | acicular     | DED    | laser=350W | UTS=939MPa
  [2] sim=0.2436 | acicular     | DED    | laser=70W | UTS=795MPa
  [3] sim=0.2436 | acicular     | LPBF   | laser=400W | UTS=940MPa
  [4] sim=0.2435 | acicular     | LPBF   | laser=320W | UTS=450MPa
  [5] sim=0.2430 | acicular     | DED    | UTS=895MPa
  Time: 5.6s

── STEP 3: Diffusion ────────────────────────────────────
[Diffusion] Loading SD 1.5 + LoRA...
Loading weights: 100%|█████████████████████████████████████████████████████| 196/196 [00:00<00:00, 211.28it/s]
CLIPTextModel LOAD REPORT from: /scratch/kishord/llm/base_models/stable_diffusion_v1_5/text_encoder128.43it/s]
Key                                | Status     |  | 
-----------------------------------+------------+--+-
text_model.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading pipeline components...: 100%|███████████████████████████████████████████| 6/6 [00:02<00:00,  2.22it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
[Diffusion] Ready.
  [v5] Seed override: base=200, seeds=[200, 217, 234]
  [v5] No hint for ('WAAM', 'pearlitic_steel', '10μm'), using visual_desc
Token indices sequence length is longer than the specified maximum sequence length for this model (82 > 77). Running this sequence through the model will result in indexing errors
  [WARN] Prompt trimmed to 77 tokens
  [PROMPT] acicular microstructure, WAAM, SEM image, additive manufacturing, scale_bar_10um, field_of_view_10um, high_magnification, The microstructure exhibits a columnar grain structure with well-defined boundaries, indicative of directional solidification. The texture suggests a preferred orientation of grains along the build direction, which is characteristic
100%|█████████████████████████████████████████████████████████████████████████| 30/30 [02:13<00:00,  4.43s/it]
    Saved 10μm → /scratch/kishord/multimodal/MatCLIP/pipeline_outputs_v5/waam_pearlsteel_vlasov2026_20260623_113435/waam_pearlsteel_vlasov2026_20260623_113435_10um_s200.png
  [v5] No hint for ('WAAM', 'pearlitic_steel', '20μm'), using visual_desc
  [WARN] Prompt trimmed to 76 tokens
  [PROMPT] acicular microstructure, WAAM, SEM image, additive manufacturing, scale_bar_20um, field_of_view_32um, medium_high_magnification, The microstructure exhibits a columnar grain structure with well-defined boundaries, indicative of directional solidification. The texture suggests a preferred orientation of grains along the build direction,
100%|█████████████████████████████████████████████████████████████████████████| 30/30 [01:00<00:00,  2.01s/it]
    Saved 20μm → /scratch/kishord/multimodal/MatCLIP/pipeline_outputs_v5/waam_pearlsteel_vlasov2026_20260623_113435/waam_pearlsteel_vlasov2026_20260623_113435_20um_s217.png
  [v5] No hint for ('WAAM', 'pearlitic_steel', '50μm'), using visual_desc
  [WARN] Prompt trimmed to 77 tokens
  [PROMPT] acicular microstructure, WAAM, SEM image, additive manufacturing, scale_bar_50um, field_of_view_64um, medium_magnification, The microstructure exhibits a columnar grain structure with well-defined boundaries,indicative of directional solidification. The texture suggests a preferred orientation of grains along the build direction, which is characteristic
100%|█████████████████████████████████████████████████████████████████████████| 30/30 [00:58<00:00,  1.96s/it]
    Saved 50μm → /scratch/kishord/multimodal/MatCLIP/pipeline_outputs_v5/waam_pearlsteel_vlasov2026_20260623_113435/waam_pearlsteel_vlasov2026_20260623_113435_50um_s234.png
  ✓ [ACICULAR  ] [10um]
  ✓ [ACICULAR  ] [20um]
  ✓ [ACICULAR  ] [50um]

[GRID] Saved → /scratch/kishord/multimodal/MatCLIP/pipeline_outputs_v5/waam_pearlsteel_vlasov2026_20260623_113435/waam_pearlsteel_vlasov2026_20260623_113435_grid.png
  Time: 287.8s

============================================================
  DONE
============================================================
  Morphology         : ACICULAR
  Process type       : WAAM

  Property predictions:
    Laser power (W)             :    285.0 ±  127.4  (n=4)
    Scan speed (mm/s)           :    666.7 ±  411.0  (n=3)
    Hatch spacing (μm)          :    100.0 ±    0.0  (n=1)
    Layer height (μm)           :     35.0 ±    5.0  (n=2)
    UTS (MPa)                   :    803.9 ±  184.7  (n=5)
    Yield strength (MPa)        :    636.8 ±  278.6  (n=4)
    Hardness (HV)               :    348.7 ±   48.7  (n=2)
    Elongation (%)              :     25.0 ±    7.0  (n=2)
    Grain size (μm)             :     40.0 ±    0.0  (n=1)

  Output image → /scratch/kishord/multimodal/MatCLIP/pipeline_outputs_v5/waam_pearlsteel_vlasov2026_20260623_113435/waam_pearlsteel_vlasov2026_20260623_113435_grid.png
  JSON         → /scratch/kishord/multimodal/MatCLIP/pipeline_outputs_v5/waam_pearlsteel_vlasov2026_20260623_113435/waam_pearlsteel_vlasov2026_20260623_113435_result.json


[ERROR] Could not find 20um generated image in /scratch/kishord/multimodal/MatCLIP/pipeline_outputs_v6/waam_pearlsteel_vlasov2026_20260623_113435/
```
