
import os, sys, json

os.environ["TORCHAUDIO_USE_BACKEND"] = "ffmpeg"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open(sys.argv[1]) as f:
    p = json.load(f)

sys.path.insert(0, p["ace_step_dir"])

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.gpu_config import get_gpu_config, set_global_gpu_config
from acestep.model_downloader import ensure_lm_model

gpu_config = get_gpu_config()
set_global_gpu_config(gpu_config)

checkpoint_dir = os.path.join(p["ace_step_dir"], "checkpoints")

dit_handler = AceStepHandler()
_, success = dit_handler.initialize_service(
    project_root=p["ace_step_dir"], config_path="acestep-v15-turbo",
    device="auto", use_flash_attention=False, compile_model=False,
    offload_to_cpu=False, offload_dit_to_cpu=False,
    quantization=None, use_mlx_dit=True,
)
assert success, "DiT init failed"

llm_handler = LLMHandler()
try:
    ensure_lm_model(model_name="acestep-5Hz-lm-1.7B", checkpoints_dir=checkpoint_dir)
except:
    pass

_, lm_ok = llm_handler.initialize(
    checkpoint_dir=checkpoint_dir, lm_model_path="acestep-5Hz-lm-1.7B",
    backend="pt", device="auto", offload_to_cpu=False, dtype=None,
)

params = GenerationParams(
    task_type="text2music",
    caption=p["caption"],
    lyrics=p["lyrics"],
    vocal_language="en", bpm=p["bpm"], keyscale=p["key"],
    duration=p["duration"], inference_steps=8, shift=3.0,
    seed=p["seed"], thinking=lm_ok,
    use_cot_metas=True, use_cot_caption=True, use_cot_language=True,
)
config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[p["seed"]], audio_format="wav")

result = generate_music(
    dit_handler=dit_handler, llm_handler=llm_handler,
    params=params, config=config, save_dir=p["output_dir"],
)

if result.success and result.audios:
    print("OUTPUT_PATH:" + result.audios[0].get("path", ""))
else:
    print("GENERATION_FAILED:" + str(getattr(result, "error", "unknown")))
