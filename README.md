# trainer
A library to implement state of the art algorithms for foundational models

You can run the examples in `trainer/examples/` with:

```bash
bash <script-path>
```

For `trainer/examples/argilla_mix_7k.sh`, different algorithms are supported via `--algorithm` (or `-a`):

| Algorithm | Command |
|-----------|---------|
| SFT | `bash trainer/examples/gsm8k_sft.sh` |
| DPO | `bash trainer/examples/argilla_mix_7k.sh --algorithm dpo` |
| ORPO | `bash trainer/examples/argilla_mix_7k.sh --algorithm orpo` |
| SimPO | `bash trainer/examples/argilla_mix_7k.sh --algorithm simpo` |
| APO (zero) | `bash trainer/examples/argilla_mix_7k.sh --algorithm apo_zero` |
| APO (down) | `bash trainer/examples/argilla_mix_7k.sh --algorithm apo_down` |
| KTO | `bash trainer/examples/argilla_mix_7k.sh --algorithm kto` |

**Note for KTO:** Preferences are automatically labeled as 1 (chosen) and 0 (rejected).

```bibtex
@misc{Tan2025RL2,
    author={Chenmien Tan and Simon Yu and Lanbo Lin and Ze Zhang and Yuanwu Xu and Chenhao Jiang and Tianyuan Yang and Sicong Xie and Guannan Zhang},
    title={RL2: Ray Less Reinforcement Learning},
    note={GitHub repository},
    howpublished={\url{https://github.com/ChenmienTan/RL2}},
    year={2025}
}
```