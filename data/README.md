# data/

원본 데이터는 git에 커밋하지 않는다. 실제 데이터는 `$AUDIO_SAFETY_DATA_DIR`
(기본: `/workspace/audio_safety/data`)에 두고, 이 디렉터리에는 설명과 준비 스크립트 산출 메타데이터만 둔다.

## 필요한 데이터 (design.md §2)

### Cone 구성용 (text)

| Source | 용도 | 비고 |
|---|---|---|
| AdvBench / HarmBench / SORRY-Bench | 카테고리 라벨 있는 harmful text | 6개 harm 카테고리, 카테고리당 256–512쌍 |
| XSTest | borderline-safe benign | over-refusal 경계 — refusal 축(주제 탐지 아님)을 잡기 위해 필수 |
| Alpaca / just-eval | 일반 benign | 주제 분포를 harmful과 매칭 |

### Drift probe용 (audio)

| Family | 렌더링 | 비고 |
|---|---|---|
| plain | 표준 TTS (또는 `WeifeiJin/AdvBench-Audio` 재사용) | |
| nonspeech | plain + 환경음 prepend/append/mix | |
| style | emotion/accent/age/gender/rate 변형 | voice seed ≥3개로 분산 |
| perturbed | AJailBench Audio Perturbation Toolkit | AdvWave식 gradient 최적화는 exp1 제외 |

## 디렉터리 규약 (workspace 하위)

```
$AUDIO_SAFETY_DATA_DIR/
  text/
    harmful/<category>.jsonl
    benign/{xstest,alpaca}.jsonl
  audio/
    <family>/<content_id>[_seed<k>].wav
  manifests/
    drift_contents.jsonl        # 선정된 150개 내용 + 카테고리
    comprehension_results.jsonl # 전사 필터 결과 (통과/탈락 + 사유)
```
