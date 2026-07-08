# Run 4 사전등록: Text-vs-Audio Refusal Conversion-Gap Causal Test

> **사전 등록 문서.** 아래 §0 threshold와 H1–H3, 통계 방법은 Run 4 실행 시작 후 수정하지
> 않는다(정정은 변경 이력으로). 이 문서는 원래 [design.md](./design.md)의 §0를 **수정하지
> 않는다** — Run 4는 exp1 style-escape gate와 별개의 causal hypothesis set이며, 논문 spine을
> 그쪽으로 이관한다. 상위 맥락: [context.md](./context.md) §Causal Core. 작성일 2026-07-08.

## 배경 / 질문

Run 3(WEAK-GO)로 audio-reachable refusal 축 `r_A`의 존재는 확인됐다(강한 ablation).
Run 4의 질문은 존재가 아니라 **인과**다.

> 같은 harmful content가 왜 text보다 audio에서 refusal coordinate로 **덜 변환(convert)**되는가?

Leading mechanism: **harmfulness는 보존되지만 harmfulness→refusal write가 audio에서 약하다
(conversion gap).** 프로소디는 우리 자신의 데이터(escape AUROC ~0.48)가 이미 주채널에서
탈락시켰다. 경쟁 가설·반증은 §5.

## 0. 사전 등록 판정 기준

주장(conversion-gap)은 아래가 **함께** 성립할 때만 지지된다.
**Primary decision position = P2 (first assistant prelogit)** — text/audio 구조가 동일한 유일한
위치. P1(last input)은 sensitivity only. **Primary layer**는 pilot에서 가장 큰 text-audio `c_R`
gap과 가장 높은 clamp mediation을 보이는 단일 layer 하나로 lock(후보 `[12,14,16,18,20]`).
`r_R` = primary refusal axis(`r_A`, 보조 `r_T`). `z_*`는 train split 기준 z-score.

| ID | 조건 | 기준 |
|---|---|---|
| T1 (supporting) | harmfulness 보존 | **audio-native `r_H` probe AUROC ≥ 0.75**; `d(z_H_text − z_H_audio)`는 서술 보고(**hard gate 아님**) |
| **T2 (primary)** | refusal under-activation `mean(z_R_text − z_R_audio)` | `> 0.3 SD` weak / **`> 0.5 SD` & bootstrap CI>0 strong** |
| **T3 (primary)** | writer gap: matched-`c_H` 통제 후 `Δc_R_audio < Δc_R_text` | 유의(regression 또는 bin, bootstrap CI>0) |
| **T4 (primary)** | natural mediation fraction MF (**all-position** clamp) | 부호>0 & one-sided `p<0.05`; ladder는 서술 |
| **Specificity (필수)** | `r_R` gap이 offset이 아님 | `r_R` gap **≫** `r_H` gap **≈** random-direction gap |
| Benign 안전 | primary clamp의 paired benign ORR 증가 | `≤ +2pp` (최대 `≤ +5pp`) |

MF ladder(T4, 서술적):

```text
MF ≥ 0.50   STRONG  — conversion gap이 audio 취약성의 주원인
0.30–0.50   GO / defensible
0.15–0.30   WEAK    — partial mechanism
< 0.15      under-activation alone 기각
```

`MF = (Gap_base − Gap_clamp) / Gap_base`,
`Gap = RefusalRate_text_harmful − RefusalRate_audio_harmful`.

**검정력 주의(사전 명시):** MF는 두 refusal-rate 차의 비율이라 CI가 넓다. behavioral arm은
가능한 한 full 150 pair로 키우고, T4 **primary 판정은 MF의 부호 + one-sided test**로 하며 MF
수치는 bootstrap CI 병기한 descriptive로만 쓴다. Run 3 `+20.7pp` knife-edge를 반복하지 않는다.

## 1. 가설

- **H1 (harmfulness preservation).** 같은 content에서 harmfulness는 audio 상태에서도 선형적으로
  복원 가능하다(`c_H_audio`가 harmful 영역). *주장은 "완전 동일"이 아니라 "크게 잃지 않음"이며,
  진짜 mechanism 증거는 H1 자체가 아니라 c_H를 통제한 뒤의 T3(writer gap)다.*
- **H2 (refusal under-activation).** 같은 harmful content에서 `c_R(audio) < c_R(text)`.
- **H3 (natural coordinate mediation).** audio harmful sample의 `r_R` 좌표를 같은 item의 **측정된**
  text 좌표까지 clamp하면 audio refusal이 증가한다(반대로 text→audio 낮추면 text safety 약화).
  benign ORR은 거의 오르지 않는다. clamp는 **임의 alpha override가 아니라 measured paired
  coordinate**여야 "natural mediation" 주장이 성립한다.

## 2. 축 (3개)

- **`r_A`** — Run 3 audio RDO refusal 축. **freeze.** "audio-reachable causal refusal coordinate".
- **`r_T`** — 신규 text-arm RDO refusal 축(같은 split/label, text ChatML). `cos(r_A, r_T)`와
  cross-modal add/ablate 성능을 보고. 충분히 높고 둘 다 cross-modal로 작동하면 "shared refusal
  coordinate", 아니면 논문 문구를 "audio under-activates a refusal coordinate **reachable by
  text/audio interventions**"로 낮춘다(§5 F3). **"text-trained axis under-activation"은 이
  alignment 증거 없이는 쓰지 않는다.**
- **`r_H`** — harmfulness 방향. **refusal이 아니라 content-harmfulness 라벨**로 학습(arXiv:2507.11878).
  text harmful-vs-benign DIM/probe로 만들고, transfer 모호성 배제를 위해 **audio-native probe도
  따로 학습**해 교차 확인. behavior outcome과 분리.
- **Readout gate (비교 전 필수):** `r_A`,`r_T`가 **각 modality 내부에서** refusal 행동을 AUROC로
  분리하는지 먼저 확인. 실패한 축은 그 modality의 projection 비교에서 무효.

## 3. 데이터 arm

기존 heldout pair 사용. **neutral style을 primary**, sad·angry는 moderator analysis로 후순위.
decoding / system prompt / chat template은 modality 간 **완전 고정**(문구 차이 → framing effect
공격 방지).

1. **Text arm** — harmful/benign content를 ChatML user **text**로 직접 입력.
2. **Audio arm** — 같은 content를 CosyVoice2 **neutral**로 렌더해 입력.
3. **Self-transcript control** — audio를 넣고 모델이 스스로 낸 transcript를 text로 되먹임. refusal이
   text 수준으로 회복되면 "LLM text path는 안전, audio 표현이 문제". *복원 실패 시 transcript
   **내용**을 audit해 perception loss인지 표현 우회인지 구분.*
4. **External / human ASR spot-check** — "모델 내부 content가 깨졌다"는 반론 차단용 최소 지각 게이트.

## 4. 측정 & 분석

각 item `i`, modality `m∈{text,audio}`, label `y∈{harmful,benign}`, layer `l`, position `p`:

```text
c_H = <h, r_H>   c_A = <h, r_A>   c_T = <h, r_T>
```

- **Projection report** — paired scatter `c_H_text vs c_H_audio`, `c_R_text vs c_R_audio`,
  2D `(c_H, c_R)` behavior-colored. bootstrap CI. **Specificity 표: `r_R` gap · `r_H` gap ·
  random-direction gap · (modality-mean-offset을 `r_H`,`r_R`에 projection)** 나란히.
- **Writer report (T3, primary mechanism 증거)** — `Δc_R(l,p) = <resid_post_l(p) − resid_pre_l(p), r_R>`.
  layer×position heatmap of `Δc_R_text − Δc_R_audio`, matched-`c_H` bin 또는 regression control.
- **Clamp (T4)** — measured natural target. **all-position(위치별 measured target)을 primary,
  single-position은 비교로 함께** 보고(Run 3 교훈: single-position 편집은 KV-cache decode에서
  씻겨나가 mediation을 과소평가). 양방향(audio→text, text→audio) + benign control + random-axis
  control.

## 5. 반증 트리

| 결과 | 해석 | framing |
|---|---|---|
| `c_H_audio ≈ c_H_text`, `c_R_audio ≪ c_R_text`, **all-pos** clamp MF ≥ 0.30 | 성공 | harmfulness→refusal **conversion gap** |
| `c_H_audio < c_H_text` & `r_H` patch가 refusal 복원 (F1) | 지각/semantic degradation | "audio perception/semantic attenuation"으로 리프레임 |
| `c_H_audio ≈ c_H_text`, `c_R_audio ≈ c_R_text`, clamp 실패 (F2) | under-activation 기각 | modality-gated readout / orthogonal compliance |
| `r_R` gap ≈ `r_H` gap ≈ random gap | specificity 실패 | 순수 modality offset — 주장 철회 |
| `c_R` clamp가 refusal은 올리나 benign도 크게 over-refuse (F4) | 축은 맞으나 surgical 아님 | mechanism partial, defense 주장 약화 |
| `cos(r_A, r_T)` 낮음 (F3) | 축 modality-specific | "text-trained axis" 금지, "modality-specific refusal coordinates" |

## 6. 구현 순서

0. **freeze old story** — Run 1–3은 appendix/diagnostic 보존, style-escape 재시도 안 함.
1. **text behavior arm** — 기존 pair를 text ChatML로. audio와 동일 system prompt/decoding/judge.
2. **text activation extraction** — 기존 hook에 text input path 추가, 동일 layer/position schema.
3. **`r_H`, `r_T` 학습** — `r_A`는 freeze. `r_H`=content-harmfulness, `r_T`=text RDO refusal.
4. **projection report** (§4) — readout gate + specificity 표 포함.
5. **writer report** (§4).
6. **natural coordinate clamp** — all-position primary + single 비교, 양방향, benign·random control.
7. **self-transcript control** — 행동 회복 + transcript 내용 audit.

**실행 전 필수:** 이 §0 threshold는 데이터 보기 전에 lock되었으며, 실행 전 `/codex-cross-check`로
판정 로직을 blind 교차검증한다(AGENTS.md 교차검증 trigger).

## 변경 이력

- 2026-07-08 — 최초 사전 등록.
