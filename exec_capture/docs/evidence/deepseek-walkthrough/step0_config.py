import json
from huggingface_hub import hf_hub_download
path = hf_hub_download("deepseek-ai/DeepSeek-V3", "config.json")
cfg = json.load(open(path))
print("downloaded:", path.split("/")[-1], "| fields:", len(cfg))
for k in ["architectures","model_type","hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads",
          "q_lora_rank","kv_lora_rank","qk_nope_head_dim","qk_rope_head_dim","v_head_dim",
          "intermediate_size","moe_intermediate_size","first_k_dense_replace","n_routed_experts","n_shared_experts",
          "num_experts_per_tok","n_group","topk_group","routed_scaling_factor","scoring_func","norm_topk_prob",
          "num_nextn_predict_layers","vocab_size","rope_scaling","quantization_config"]:
    print(f"  {k:24s} = {cfg.get(k, '<absent>')}")
