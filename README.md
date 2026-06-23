# multi_modal

#kishord account paramrudra.iitb

directory -- ```/scratch/kishord/multimodal/MatCLIP```

env activation -- ```source matclip_env/bin/activate```

command to run model imange generation and validation -- ```python3 matclip_generate_and_validate.py         --params "WAAM, pearlitic steel, current 118A, travel speed 350mm/min"         --reference vlasov_fig6a_ref.png         --label "WAAM PearlSteel Vlasov2026"         --seed 200```

the observation --


1. The initial model from the directory is the waam_hybrid model and the diffusion model and now we have downloaded the mistrial and diffusion model to our directory and using that path

2. Once the model weights are loaded, it ia taking lot of time to  at the step ```[RAG] Load failed: [Errno 2] No such file or directory: '/scratch/kishord/multimodal/MatCLIP/rag_store/bm25_index.pkl' — will skip RAG context```.

