# data/

Raw data is not committed. Store actual files under `$AUDIO_SAFETY_DATA_DIR`
(default: `/workspace/audio_safety/data`). This repository directory only records
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

## Audio Rendering

TTS engine:

```text
CosyVoice2
```

Initial styles:

```text
neutral
sad
fearful
angry
elderly_male
child_female
```

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

Geometry analysis may only use rows satisfying:

```text
WER <= 5%
core_tokens_preserved == true
style_passed == true
duration not an outlier
```

## Optional Follow-up Sources

Only use these after the first gate if FigStep/SafeBench lacks enough valid pairs:

| Source | Role |
|---|---|
| HarmBench behaviors | broader harmful behavior taxonomy for robustness |
| Do-Not-Answer | refusal-oriented prompts for auxiliary validation |
| XSTest | borderline benign prompts to stress over-refusal |
