## Low Rank Adaptation Methods on Segment Anything for LIVECell

Insights:
- There's no real memory advantage actually unless it's truly scaled up. For instance:
    - `vit_b`:
        - SAM: 93M (takes ~50GB)
        - SAM-LoRA: 4.4M (takes ~61GB)
    - `vit_l`:
        - SAM: 312M (takes ~63GB)
        - SAM-LoRA: 4.4M (takes ~61GB)
    - `vit_h`:
        - SAM: 641M (takes ~73GB)
        - SAM-LoRA: 4.7M (takes ~67GB)

- Question: Would quantization lead to better results? (e.g. QLoRA) or parallel adaptation? (e.g. DoRA)
