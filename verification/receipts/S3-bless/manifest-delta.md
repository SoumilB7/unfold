# S3 approved bless — manifest-delta inspection

- prior manifest: committed tree `c47a975`
- rebuilt from: implementation tree `56d66de`
- denominator: 29 witnesses
- expected changed set: the ten witnesses authorized in `independent-verdict.md`
- actual changed set: **exactly the expected set**
- unexpected witnesses: **none**
- missing approved witnesses: **none**
- manifest version/tool record changed: **no**
- witness count changed: **no (29 → 29)**

| witness | changed canonical surfaces | distinct views | frozen fixture input changed only because its approved lock changed |
|---|---|---:|---|
| `dbrx-base` | expanded, gallery, HTML metadata, IR, views | 6 → 6 | yes |
| `deepseek-v3` | expanded, gallery, HTML metadata, IR, views | 11 → 11 | yes |
| `gemma-2-2b-it` | gallery, HTML metadata, views | 6 → 6 | yes |
| `glm-4-5` | expanded, gallery, HTML metadata, IR, views | 9 → 9 | yes |
| `gpt-oss-20b` | expanded, gallery, HTML metadata, IR, views | 8 → 8 | yes |
| `lumina-image-2-0` | gallery, HTML metadata, views | 16 → 16 | yes |
| `musicgen-small` | gallery, HTML metadata, IR, views | 8 → 8 | yes |
| `pixart-sigma-xl-2-1024-ms` | expanded, gallery, HTML metadata, IR, views | 13 → 13 | yes |
| `sana-1600m-1024px-diffusers` | gallery, HTML metadata, views | 12 → 12 | yes |
| `stable-diffusion-3-5-large` | expanded, gallery, HTML metadata, IR, views | 19 → 19 | yes |

The generated Sable-fixture diff names the same exact ten witnesses. Gallery
file changes are restricted to their reviewed MoE, sliding-window, MusicGen
source, and zero-layer-denoiser views. No other fixture or gallery changed.
