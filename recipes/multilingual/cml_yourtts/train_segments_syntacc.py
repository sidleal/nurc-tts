import os
import csv
import re
from collections import Counter
from dataclasses import dataclass, field
import unicodedata
import wave
from pathlib import Path

import torch
from trainer import Trainer, TrainerArgs
from datasets import load_dataset

from TTS.config import load_config
from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig, VitsDataset
from TTS.utils.downloaders import download_libri_tts
from torch.utils.data import DataLoader
from TTS.utils.samplers import PerfectBatchSampler
torch.set_num_threads(24)

"""
    This recipe replicates the first experiment proposed in the CML-TTS paper (https://arxiv.org/abs/2306.10097). It uses the YourTTS model.
    YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
"""
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Name of the run for the Trainer
RUN_NAME = "YourTTS-Syntacc-PT_NURC"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
# OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")  # "/raid/coqui/Checkpoints/original-YourTTS/"
OUT_PATH = "/opt/training/syntacc/TTS/NURC/runs"
# If you want to do transfer learning and speedup your training you can set here the path to the CML-TTS available checkpoint that cam be downloaded here:  https://drive.google.com/u/2/uc?id=1yDCSJ1pFZQTHhL09GMbOrdjcPULApa0p
RESTORE_PATH = "/app/checkpoints_yourtts_cml_tts_dataset/best_model.pth"  # Download the checkpoint here:  https://drive.google.com/u/2/uc?id=1yDCSJ1pFZQTHhL09GMbOrdjcPULApa0p

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 6

SAMPLE_RATE = 24000


DASHBOARD_LOGGER="tensorboard"
LOGGER_URI = None 

# DASHBOARD_LOGGER = "clearml"
# LOGGER_URI = "s3://coqui-ai-models/TTS/Checkpoints/YourTTS/NURC/"



# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
# MAX_AUDIO_LEN_IN_SECONDS = float("inf")
MAX_AUDIO_LEN_IN_SECONDS = 15
# Hugging Face dataset setup (nilc-nlp/nurc_tts)
USE_HF_NURC_DATASET = True
HF_DATASET_ID = "sidleal/nurc_tts_24khz"
#HF_DATASET_ID = "nilc-nlp/nurc_tts"
HF_ACCENT_SPLITS = ("sao_paulo", "recife")
HF_LOCAL_DATA_ROOT = "/opt/training/syntacc/datasets/nurc_tts"
# HF_LOCAL_DATA_ROOT = os.path.join(CURRENT_PATH, "datasets", "nurc_tts")
HF_MAX_SAMPLES_PER_SPLIT = None  # Set an int for quick debugging (e.g., 2000)
HF_TOP_SPEAKERS_PER_SPLIT = 50  
HF_EXCLUDED_INQUIRIES = {
    "sao_paulo": {
        "SP_DID_052",
        "SP_DID_065",
        "SP_DID_102",
        "SP_DID_110",
        "SP_DID_014",
        "SP_DID_023",
        "SP_DID_027",
        "SP_DID_031",
    },
    "recife": {
        "RE_DID_001",
        "RE_DID_013",
        "RE_DID_025",
        "RE_DID_037",
        "RE_DID_045",
        "RE_DID_044",
        "RE_DID_032",
        "RE_DID_058",
    },
}
HF_FORCE_REBUILD_METADATA = True  # Keep True to ensure inquiry filters are always applied.
os.makedirs(HF_LOCAL_DATA_ROOT, exist_ok=True)

@dataclass
class TrainSyntaccArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={"help": "Path to the config file."})


def _run_from_config_path_if_provided() -> bool:
    
    train_args = TrainSyntaccArgs()
    parser = train_args.init_argparse(arg_prefix="")
    args, config_overrides = parser.parse_known_args()
    train_args.parse_args(args)

    if not args.config_path:
        print(">>> EXECUTION MODE: script defaults (no --config_path).")
        return False

    config = load_config(args.config_path)
    if len(config_overrides) > 0:
        config.parse_known_args(config_overrides, relaxed_parser=True)

    resolved_run_path = os.path.join(config.output_path, config.run_name)
    print(">>> EXECUTION MODE: --config_path provided.")
    print(f">>> config_path={args.config_path}")
    print(f">>> effective output_path={config.output_path}")
    print(f">>> expected run directory={resolved_run_path}")

    for dataset_conf in config.datasets:
        dataset_path = Path(dataset_conf.path)
        metadata_file = dataset_path / dataset_conf.meta_file_train
        if metadata_file.exists():
            continue

        if dataset_conf.formatter == "coqui" and dataset_conf.dataset_name == "nurc_tts":
            split_name = dataset_conf.language or dataset_path.name
            output_root = str(dataset_path.parent)
            print(
                f">>> Missing metadata for split '{split_name}'. "
                f"Preparing dataset at {output_root} ..."
            )
            prepare_nurc_tts_split(
                split_name=split_name,
                output_root=output_root,
                max_samples=HF_MAX_SAMPLES_PER_SPLIT,
            )

    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = setup_model(config, train_samples + eval_samples)
    trainer = Trainer(
        train_args,
        model.config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    print(">>> TRAINING OVERVIEW")
    print(f">>> run_name={config.run_name}")
    print(f">>> output_path={config.output_path}")
    print(f">>> train_samples={len(train_samples)} eval_samples={len(eval_samples)}")
    print(
        f">>> batch_size(train/eval)={config.batch_size}/{config.eval_batch_size} "
        f"print_step={config.print_step} save_step={config.save_step}"
    )
    print(">>> Starting trainer.fit()")
    try:
        trainer.fit()
        print(">>> trainer.fit() finished successfully")
    except Exception as exc:
        print(f">>> trainer.fit() stopped with exception: {type(exc).__name__}: {exc}")
        raise
    return True



def _slugify_filename(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("_")


def _sanitize_speaker_name(speaker_name: str) -> str:
    safe = _slugify_filename((speaker_name or "unknown").replace(" ", "_"))
    return safe or "unknown"


def _normalize_metadata_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("|", " ")).strip()


def _build_speaker_name(inquiry_id: str, raw_speaker_name: str) -> str:
    combined_speaker_name = f"{inquiry_id}_{raw_speaker_name}" if inquiry_id else raw_speaker_name
    return _sanitize_speaker_name(combined_speaker_name)


def _select_top_speakers_for_split(dataset, split_name: str, excluded_inquiries: set, top_k: int):
    speaker_counts = Counter()
    skipped_inquiry = 0
    skipped_invalid = 0

    for item in dataset:
        inquiry_id = (item.get("inquiry") or "").strip().upper()
        if inquiry_id and inquiry_id in excluded_inquiries:
            skipped_inquiry += 1
            continue

        text = _normalize_metadata_text(item.get("text") or "")
        audio = item.get("audio")
        if not text or audio is None:
            skipped_invalid += 1
            continue

        raw_speaker_name = (item.get("speaker") or "unknown").strip()
        print(f">>>AQUI ESTA O RAW_SPEAKER_NAME: {raw_speaker_name}")
        print(f">>>AQUI ESTA O INQUIRY_ID: {inquiry_id}")
        speaker_name = _build_speaker_name(inquiry_id, raw_speaker_name)
        print(f">>>AQUI ESTA O SPEAKER_NAME: {speaker_name}")
        speaker_counts[speaker_name] += 1

    ranked = sorted(speaker_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is None or top_k <= 0:
        selected_ranked = ranked
    else:
        selected_ranked = ranked[:top_k]
    selected_speakers = {speaker for speaker, _ in selected_ranked}

    print(
        f">>> Split '{split_name}': found {len(ranked)} speakers after base filtering "
        f"(skipped inquiry={skipped_inquiry}, invalid={skipped_invalid})."
    )
    if top_k is not None and top_k > 0:
        print(
            f">>> Split '{split_name}': selecting top {len(selected_ranked)} speakers by segment count."
        )
    preview = selected_ranked[:10]
    if preview:
        preview_str = ", ".join([f"{speaker}:{count}" for speaker, count in preview])
        print(f">>> Split '{split_name}': top speakers preview => {preview_str}")

    return selected_speakers


def _write_wav_int16(file_path: Path, audio_array, sample_rate: int) -> None:
    waveform = torch.tensor(audio_array, dtype=torch.float32).clamp(-1.0, 1.0)
    pcm = (waveform * 32767.0).to(torch.int16).numpy().tobytes()
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def _decode_audio_or_none(audio, split_name: str, index: int, segment_name: str):
    try:
        audio_array = audio["array"]
        sample_rate = int(audio["sampling_rate"])
    except Exception as exc:
        print(
            f">>>DECODE ERROR !O!O!\O/|>> Skipping sample {split_name}[{index}] ({segment_name}) due to audio decode error: "
            f"{type(exc).__name__}: {exc}"
        )
        return None, None

    if audio_array is None or len(audio_array) == 0:
        print(f">>> Skipping sample {split_name}[{index}] ({segment_name}) because audio is empty.")
        return None, None

    return audio_array, sample_rate


def prepare_nurc_tts_split(split_name: str, output_root: str, max_samples: int = None):
    split_root = Path(output_root) / split_name
    wav_root = split_root / "wavs"
    metadata_path = split_root / f"metadata_coqui_{split_name}.csv"
    split_root.mkdir(parents=True, exist_ok=True)
    wav_root.mkdir(parents=True, exist_ok=True)

    sample_speaker = None
    excluded_inquiries = {item.strip().upper() for item in HF_EXCLUDED_INQUIRIES.get(split_name, set())}
    rebuild_metadata = HF_FORCE_REBUILD_METADATA or (not metadata_path.exists())
    if rebuild_metadata:
        print(f">>> Preparing Hugging Face split '{split_name}' from {HF_DATASET_ID}")
        dataset = load_dataset(HF_DATASET_ID, split=split_name)
        selected_speakers = _select_top_speakers_for_split(
            dataset=dataset,
            split_name=split_name,
            excluded_inquiries=excluded_inquiries,
            top_k=HF_TOP_SPEAKERS_PER_SPLIT,
        )
        with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
            writer = csv.writer(metadata_file, delimiter="|")
            writer.writerow(["audio_file", "text", "speaker_name"])
            total_written = 0
            total_skipped_inquiry = 0
            total_skipped_audio = 0
            total_skipped_speaker_filter = 0
            for index, item in enumerate(dataset):
                if max_samples is not None and total_written >= max_samples:
                    break

                inquiry_id = (item.get("inquiry") or "").strip().upper()
                if inquiry_id and inquiry_id in excluded_inquiries:
                    total_skipped_inquiry += 1
                    continue

                text = _normalize_metadata_text(item.get("text") or "")
                audio = item.get("audio")
                if not text or audio is None:
                    continue

                raw_speaker_name = (item.get("speaker") or "unknown").strip()
                speaker_name = _build_speaker_name(inquiry_id, raw_speaker_name)
                if speaker_name not in selected_speakers:
                    total_skipped_speaker_filter += 1
                    continue
                print(
                    f">>>AQUI ESTA O SPEAKER_NAME: inquiry={inquiry_id or 'unknown'} "
                    f"speaker={raw_speaker_name} -> {speaker_name}"
                )
                if sample_speaker is None:
                    sample_speaker = speaker_name

                segment_name = _slugify_filename(item.get("segment", f"{split_name}_{index}.wav"))
                if not segment_name.endswith(".wav"):
                    segment_name = f"{segment_name}.wav"
                file_name = f"{index:09d}_{segment_name}"
                rel_audio_path = f"wavs/{file_name}"
                abs_audio_path = split_root / rel_audio_path

                audio_array, sample_rate = _decode_audio_or_none(audio, split_name, index, file_name)
                if audio_array is None:
                    total_skipped_audio += 1
                    continue

                if not abs_audio_path.exists():
                    _write_wav_int16(abs_audio_path, audio_array, sample_rate)

                writer.writerow([rel_audio_path, text, speaker_name])
                total_written += 1

        print(
            f">>> Wrote {total_written} samples for split '{split_name}' at {split_root} "
            f"(skipped {total_skipped_inquiry} excluded inquiry samples, "
            f"{total_skipped_audio} decode-failing/empty samples, "
            f"{total_skipped_speaker_filter} samples outside top-{HF_TOP_SPEAKERS_PER_SPLIT} speakers)"
        )
    else:
        print(f">>> Reusing existing metadata for split '{split_name}' at {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file):
                if line_number == 0:
                    continue
                cols = line.strip().split("|")
                if len(cols) >= 3 and cols[2]:
                    sample_speaker = cols[2]
                    break

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="nurc_tts",
        meta_file_train=metadata_path.name,
        path=str(split_root),
        language=split_name,
    )
    return dataset_config, sample_speaker


def build_nurc_tts_configs():
    dataset_configs = []
    sample_speakers = {}
    for split_name in HF_ACCENT_SPLITS:
        split_config, sample_speaker = prepare_nurc_tts_split(
            split_name=split_name,
            output_root=HF_LOCAL_DATA_ROOT,
            max_samples=HF_MAX_SAMPLES_PER_SPLIT,
        )
        dataset_configs.append(split_config)
        sample_speakers[split_name] = sample_speaker or "unknown"
    return dataset_configs, sample_speakers


def build_test_sentences(dataset_configs, num_per_split: int = 3):
    test_sentences = []
    for dataset_conf in dataset_configs:
        split_name = dataset_conf.language
        #print(f">>>AQUI ESTA O SPLIT_NAME: {split_name}")
        metadata_path = Path(dataset_conf.path) / dataset_conf.meta_file_train
        #print(f">>>AQUI ESTA O METADATA_PATH: {metadata_path}")
        seen_speakers = {}  # speaker_name -> text
        #print(f">>>AQUI ESTA O SEEN_SPEAKERS: {seen_speakers}")
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                #print(f">>>AQUI ESTA O READER: {reader}")
                next(reader, None)  # skip header
                #print(f">>>AQUI ESTA O NEXT: {next(reader, None)}")
                for row in reader:
                    if len(row) < 3:
                        continue
                    _, text, speaker_name = row[0], row[1].strip(), row[2].strip()
                    #print(f">>>AQUI ESTA O TEXT: {text}")
                    #print(f">>>AQUI ESTA O SPEAKER_NAME: {speaker_name}")
                    if speaker_name and text and speaker_name not in seen_speakers:
                        seen_speakers[speaker_name] = text
                        #print(f">>>AQUI ESTA O SEEN_SPEAKERS: {seen_speakers}")
                    if len(seen_speakers) >= num_per_split:
                        break
        except Exception as exc:
            print(f">>> build_test_sentences: could not read {metadata_path}: {exc}")
            continue
        for speaker_name, text in seen_speakers.items():
            test_sentences.append([text, speaker_name, None, split_name])
            #print(f">>>AQUI ESTA O TEST_SENTENCES: {test_sentences}")
        print(
            f">>> build_test_sentences: split '{split_name}' -> "
            + ", ".join(seen_speakers.keys())
        )
        #print(f">>>AQUI ESTA O PRINT: {print(f">>> build_test_sentences: split '{split_name}' -> " + ", ".join(seen_speakers.keys()))}")
    return test_sentences

# Define here the datasets config
# brpb_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brpb.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brpb"
# )

# brba_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brba.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brba"
# )

# brportugal_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brportugal.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brportugal"
# )

# brsp_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brsp.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brsp"
# )

# brpe_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brpe.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brpe"
# )

# brmg_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brmg.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brmg"
# )

# brrj_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brrj.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brrj"
# )

# brce_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brce.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brce"
# )

# brrs_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brrs.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brrs"
# )

# bralemanha_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_bralemanha.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="bralemanha"
# )

# brgo_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brgo.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brgo"
# )

# bral_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_bral.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="bral"
# )

# brpr_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brpr.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brpr"
# )

# bres_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_bres.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="bres"
# )

# brpi_train_config = BaseDatasetConfig(
#     formatter="coqui",
#     dataset_name="mupe",
#     meta_file_train="metadata_coqui_brpi.csv",
#     path="/raid/datasets/MUPE/dataset/mupe/",
#     language="brpi"
# )

# bres_train_config, brpi_train_config  no files found 
if _run_from_config_path_if_provided():
    raise SystemExit(0)

if USE_HF_NURC_DATASET:
    DATASETS_CONFIG_LIST, HF_SAMPLE_SPEAKERS = build_nurc_tts_configs()
# else:
#     DATASETS_CONFIG_LIST = [brpb_train_config,brba_train_config,brportugal_train_config,brsp_train_config,brpe_train_config,brmg_train_config,brrj_train_config,brce_train_config,brrs_train_config,bralemanha_train_config,brgo_train_config,bral_train_config,brpr_train_config]
#     HF_SAMPLE_SPEAKERS = {}


### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(dataset_conf.path, f"H_ASP_speaker_embeddings_{dataset_conf.language}.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>>AQUI ESTA COMPUTANDO OS EMBEDDINGS: Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)


# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=512,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    inference_noise_scale=0.33,
    inference_noise_scale_dp=0.33,
    spec_segment_size=62,
    hidden_channels=192,
    hidden_channels_ffn_text_encoder=768,
    num_heads_text_encoder=2,
    num_layers_text_encoder=10,
    kernel_size_text_encoder=3,
    dropout_p_text_encoder=0.1,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    use_speaker_encoder_as_loss=False,
    # Useful parameters to enable multilingual training
    use_language_embedding=False,
    embedded_language_dim=4,
    use_adaptive_weight_text_encoder=True,
    use_perfect_class_batch_sampler=True,
    perfect_class_batch_sampler_key="language"
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="SYNTACC",
    run_description="""
            - YourTTS with SYNTACC text encoder
        """,
    dashboard_logger=DASHBOARD_LOGGER,
    logger_uri=LOGGER_URI,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=2,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    # target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00a1\u00a3\u00b7\u00b8\u00c0\u00c1\u00c2\u00c3\u00c4\u00c5\u00c7\u00c8\u00c9\u00ca\u00cb\u00cc\u00cd\u00ce\u00cf\u00d1\u00d2\u00d3\u00d4\u00d5\u00d6\u00d9\u00da\u00db\u00dc\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u0101\u0104\u0105\u0106\u0107\u010b\u0119\u0141\u0142\u0143\u0144\u0152\u0153\u015a\u015b\u0161\u0178\u0179\u017a\u017b\u017c\u020e\u04e7\u05c2\u1b20",
        punctuations="\u2014!'(),-.:;?\u00bf ",
        phonemes="iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u025a\u02de\u026b'\u0303' ",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=None,
    precompute_num_workers=3,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=True,
    test_sentences=build_test_sentences(DATASETS_CONFIG_LIST, num_per_split=3),
 
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    # weighted_sampler_attrs={"language": 1.0, "speaker_name": 1.0},
    weighted_sampler_attrs={"language": 1.0},
    weighted_sampler_multipliers={
        # "speaker_name": {
        # you can force the batching scheme to give a higher weight to a certain speaker and then this speaker will appears more frequently on the batch.
        # It will speedup the speaker adaptation process. Considering the CML train dataset and "new_speaker" as the speaker name of the speaker that you want to adapt.
        # The line above will make the balancer consider the "new_speaker" as 106 speakers so 1/4 of the number of speakers present on CML dataset.
        # 'new_speaker': 106, # (CML tot. train speaker)/4 = (424/4) = 106
        # }
    },
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the YourTTS paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH, start_with_eval=True),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
print(">>> TRAINING OVERVIEW")
print(f">>>AQUI ESTA O RUN_NAME: run_name={config.run_name}")
print(f">>>AQUI ESTA O OUTPUT_PATH: output_path={OUT_PATH}")
print(">>> EXECUTION MODE: script defaults (without --config_path).")
print(f">>>\OOOO/ effective output_path={OUT_PATH}")
print(f">>>\OOOO ESPERADO: expected run directory={os.path.join(OUT_PATH, config.run_name)}")
print(f">>>AQUI ESTA O TRAIN_SAMPLES: train_samples={len(train_samples)}")
print(f">>>AQUI ESTA O EVAL_SAMPLES: eval_samples={len(eval_samples)}")
print(
    f">>> batch_size(train/eval)={config.batch_size}/{config.eval_batch_size} "
    f"print_step={config.print_step} save_step={config.save_step}"
)
print(">>>\o/\o/ Starting trainer.fit()")
try:
    trainer.fit()
    print(">>> \O/\O/ trainer.fit() finished successfully")
except Exception as exc:
    print(f">>> DEU RUIM, trainer.fit() stopped with exception: {type(exc).__name__}: {exc}")
    raise
