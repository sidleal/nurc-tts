from pathlib import Path


CURRENT_PATH = Path(__file__).resolve().parent
SOURCE_PATH = CURRENT_PATH / "train_syntacc.py"


def _replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise RuntimeError(f"Could not find expected snippet: {old}")
    return source.replace(old, new, 1)


source = SOURCE_PATH.read_text(encoding="utf-8")
source = _replace_once(source, 'RUN_NAME = "YourTTS-Syntacc-PT_NURC"', 'RUN_NAME = "YourTTS-Syntacc-PT_NURC-10k"')
source = _replace_once(
    source,
    'HF_LOCAL_DATA_ROOT = os.path.join(CURRENT_PATH, "datasets", "nurc_tts")',
    'HF_LOCAL_DATA_ROOT = os.path.join(CURRENT_PATH, "datasets", "nurc_tts_10k")',
)
source = _replace_once(source, "HF_MAX_SAMPLES_PER_SPLIT = None", "HF_MAX_SAMPLES_PER_SPLIT = 5000")
source = _replace_once(source, "HF_FORCE_REBUILD_METADATA = True", "HF_FORCE_REBUILD_METADATA = True")

exec_globals = {
    "__name__": "__main__",
    "__file__": str(SOURCE_PATH),
}
exec(compile(source, str(SOURCE_PATH), "exec"), exec_globals)
