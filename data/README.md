# data/

Raw data is not committed. Store actual files under `$AUDIO_SAFETY_DATA_DIR`
(default: `/workspace/audio_safety_data/data`). This repository directory only records
dataset choices and manifest contracts.

## Selected Initial Dataset

Primary harmful text seed:

| Source | URL | Why |
|---|---|---|
| FigStep SafeBench | `https://github.com/CryptoAILab/FigStep` | Public harmful-question source used in the same lineage as audio safety red-teaming work; easy to render into speech with controlled TTS. |

Expected seed file after acquisition:

```text
$AUDIO_SAFETY_DATA_DIR/text/figstep/safebench.csv
```

Do **not** run geometry on this harmful-only CSV. First create a curated harmful-benign pair manifest:

```text
$AUDIO_SAFETY_DATA_DIR/text/figstep/audio_rdo_pairs.jsonl
```

Required JSONL fields:

```json
{"item_id":"...","category":"...","harmful_text":"...","benign_text":"...","source":"figstep_safebench"}
```

Pairing rule:

- keep lexical overlap high;
- change only the unsafe intent into safe handling, prevention, reporting, historical discussion, or high-level explanation;
- avoid adding jailbreak phrasing or style instructions;
- manually spot-check pairs before rendering.

Draft pair generation:

```bash
export OPENROUTER_API_KEY=<your_key>

./scripts/prepare_audio_rdo_pairs.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --limit 150
```

The OpenRouter prompt only asks for a benign control rewrite and explicitly
forbids harmful answers, steps, code, materials, or operational details. Generated
rows are marked `needs_review` by default.

## Audio Rendering

TTS engine:

```text
CosyVoice2
```

The repository provides command-template adapters instead of hardcoding a
CosyVoice2 installation. For production renders, prefer
`dataset.tts.batch_command_template`: it passes a JSONL job file to
`scripts/cosyvoice2_tts.py --batch-jsonl`, so CosyVoice2 is loaded once on the GPU
and then reused for every pending wav. `dataset.tts.command_template` remains
for single-file debugging. Templates may use:

```text
{text} {text_json} {style} {output} {item_id} {safety_label}
{batch_jsonl} {batch_jobs_file}
```

Example shape:

```bash
./scripts/render_audio_rdo.py \
  --config configs/experiments/exp1_refusal_cone_drift.yaml \
  --dry-run
```

Remove `--dry-run` only after the command template works on the GPU machine.

Initial fast-gate styles:

```text
neutral
sad
```

This is the minimum style setting used for the first direction check: neutral is
the baseline and sad is the non-neutral condition. With 150 pairs this renders
600 wav files instead of 1800. Broaden `configs/datasets/audio_rdo_gate.yaml`
back to the full style list after the RDO direction is promising.

Expected render tree:

```text
$AUDIO_SAFETY_DATA_DIR/
  audio/
    harmful/
      neutral/<item_id>.wav
      sad/<item_id>.wav
      fearful/<item_id>.wav
      angry/<item_id>.wav
      elderly_male/<item_id>.wav
      child_female/<item_id>.wav
    benign/
      neutral/<item_id>.wav
      sad/<item_id>.wav
      fearful/<item_id>.wav
      angry/<item_id>.wav
      elderly_male/<item_id>.wav
      child_female/<item_id>.wav
```

## Transcript and Style Control

Each rendered audio must have a manifest row:

```text
$AUDIO_SAFETY_DATA_DIR/manifests/audio_rdo_renders.jsonl
```

Required fields:

```json
{
  "item_id": "...",
  "safety_label": "harmful",
  "style": "neutral",
  "path": "audio/harmful/neutral/....wav",
  "transcript": "...",
  "wer": 0.0,
  "duration_s": 0.0,
  "core_tokens_preserved": true,
  "style_passed": true
}
```

ASR is intentionally simple for the first gate:

- `dataset.asr.mode: skip`: do not run ASR; pass rendered rows through with
  `transcript_control_skipped=true`. This is the current fast path.
- `dataset.asr.mode: manifest`: fill `transcript` fields in the render manifest,
  then run `./scripts/score_transcripts.py`.
- `dataset.asr.mode: command`: set `dataset.asr.command_template`; the command
  must print a transcript to stdout. The template may use `{audio}`, `{path}`,
  `{item_id}`, `{style}`, and `{safety_label}`.

Geometry analysis may only use rows satisfying:

```text
WER <= 5%
core_tokens_preserved == true
style_passed == true when style classifier is required
duration not an outlier
```

## Optional Follow-up Sources

Only use these after the first gate if FigStep/SafeBench lacks enough valid pairs:

| Source | Role |
|---|---|
| HarmBench behaviors | broader harmful behavior taxonomy for robustness |
| Do-Not-Answer | refusal-oriented prompts for auxiliary validation |
| XSTest | borderline benign prompts to stress over-refusal |
