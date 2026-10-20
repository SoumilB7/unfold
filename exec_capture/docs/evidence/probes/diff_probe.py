import torch, time, warnings, collections; warnings.filterwarnings("ignore")
from diffusers import FluxTransformer2DModel, UNet2DConditionModel, AutoencoderKL
def export(name, model, kwargs):
    t=time.time()
    try:
        n=sum(p.numel() for p in model.parameters())
        ep=torch.export.export(model,(),kwargs,strict=False)
        ops=[str(x.target).replace("aten.","").split(".")[0] for x in ep.graph.nodes if x.op=="call_function"]
        owners=collections.Counter(str(list(x.meta.get("nn_module_stack",{}).values())[-1][1]).split(".")[-1].strip("'>") if x.meta.get("nn_module_stack") else "-" for x in ep.graph.nodes if x.op=="call_function")
        print(f"OK   {name:28s} params={n:,}  {len(ops)} ops, {len(set(ops))} distinct  {time.time()-t:.1f}s  top owners={owners.most_common(4)}")
    except Exception as e:
        print(f"FAIL {name:28s} {type(e).__name__}: {str(e)[:200]}")
M="meta"
# FLUX.1-schnell transformer (full depth)
cfg=FluxTransformer2DModel.load_config("black-forest-labs/FLUX.1-schnell", subfolder="transformer")
with torch.device(M): f=FluxTransformer2DModel.from_config(cfg)
export("FLUX.1-schnell transformer", f, dict(hidden_states=torch.zeros(1,16,64,device=M), encoder_hidden_states=torch.zeros(1,8,4096,device=M),
      pooled_projections=torch.zeros(1,768,device=M), timestep=torch.zeros(1,device=M), img_ids=torch.zeros(16,3,device=M), txt_ids=torch.zeros(8,3,device=M), return_dict=False))
# SD1.5 UNet (full)
cfg=UNet2DConditionModel.load_config("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
with torch.device(M): u=UNet2DConditionModel.from_config(cfg)
export("SD1.5 UNet", u, dict(sample=torch.zeros(1,4,16,16,device=M), timestep=torch.zeros(1,device=M), encoder_hidden_states=torch.zeros(1,8,768,device=M), return_dict=False))
# SD1.5 VAE decoder path
cfg=AutoencoderKL.load_config("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
with torch.device(M): v=AutoencoderKL.from_config(cfg)
export("SD1.5 VAE", v, dict(sample=torch.zeros(1,3,32,32,device=M), return_dict=False))
