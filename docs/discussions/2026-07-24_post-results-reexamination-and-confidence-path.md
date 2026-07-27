# 실험 결과 재검토 + 확신 회복 경로 (2026-07-24)

> **성격:** 블라인드 프레이밍 단계 종료 후 **처음으로 실험 결과 파일(design.md + results.md runs 1–13 + 코드)을 읽고** 원래 가설을 재검토한 기록.
> **참여/방법:** Codex(`gpt-5.6-sol`, xhigh) 2라운드 ↔ adversarial-reviewer(독립) **수렴** + research-code-reviewer 코드 감사 + Claude 자체 문헌 web-검증.
> **이 문서는 사전등록(design.md)이 아니다.** §0은 불변. 아래 kill 기준은 *다음* 연구를 위한 제안이며 사용자 승인 전까지 확정 아님.

---

## 0. 한 줄 결론

원래 첫 가정(P1/M1/S1)은 **load-bearing 형태로 틀렸다** — 그러나 3주는 낭비가 아니다(잘못된 가설을 엄밀히 죽였고 진짜 현상 하나를 발견). 생존하는 것은 jailbreak-메커니즘 논문이 아니라 **negative/measurement 논문**(as-is 3/10, 2-model×2-TTS 재현 시 5–6). **확신 회복의 첫 수는 GPU 없이 거의 공짜인 Step 0(저장된 generation 재라벨)이고, 그 결과가 논문 vs 피벗을 결정한다.**

## 1. 확신을 잃은 이유 (실험 흐름에서 파악)

첫 가정은 세 축에서 miscalibrated였다:

| 축 | 가정 | 실제 (실험 + 문헌) |
|---|---|---|
| **위치(location)** | naturalistic content-preserving 조작(pitch/emotion/phase)이 강한 audio-specific jailbreak | 이 코너는 **약한 공격 영역**; 진짜 audio-specific 위험은 adversarial(AdvWave ASR 97.3%) / acoustic-semantic(AIA) 코너 — 이미 선점 |
| **크기(magnitude)** | audio가 text보다 **크게** 더 뚫림 | JALMBench(2505.17568): Qwen2-Audio는 text 6.9% vs audio **7.3%**(+0.4pp). 사용자 Run 4 +2.7pp n.s.와 동일 order |
| **깨끗함(cleanliness)** | 조작 가능한 단일 refusal 축 `r_A`, layer 16–18 국소 | refusal 침식은 **분산·비국소**(Run 13: σ₁/σ₈≈1.8, cross-fold≈89°); 고정 저계수 축 없음 |

**결정적 착시 (Codex):** 성공한 **trained** intervention(RDO `r_A` add +20.7pp / ablate +35.6pp = WEAK-GO)을, **부재하는 natural 현상의 대체물**로 오독했다. 같은 `r_A`가 자연 readout으로는 실패한다 — readout AUROC 0.60, style-escape AUROC 0.484(우연 이하), restoration +16.7pp(<20pp). 즉 `r_A`는 "학습된 actuator"지 "audio가 자연히 지나가는 refusal 축"이 아니다.

**체인이 끊긴 지점 = Run 4 Stage A** (matched text-audio T0): RD(audio−text) +2.7pp n.s., specificity DiD **−26.7pp**(audio 효과가 harmful-특이가 아니라 benign까지 미는 generic shift). 이후 conversion-gap → causal-attribution → emotion → phase → dissociation → factorial → rank-sweep의 모든 pivot이 같은 negative를 반복 확인했다.

## 2. "실험 결과를 의심하라"에 대한 정직한 답

**null들은 진짜다 — 코드 아티팩트가 아니다.** research-code-reviewer가 확인: 결정적 endpoint가 전부 깨진 `label_output`을 우회한다.

- Run 10 4-way(66% decoding / 21% benign / **13% genuine compliance**, ~2/246 operational) = **에이전트 직접 판정** → ROBUST
- Run 11/12 refusal = `label_output`이 아니라 "cannot engage"까지 잡는 **별도 regex** → Run 12 ROBUST
- Run 4 T0 = blinded judge + 올바른 paired McNemar/bootstrap → STOP은 아티팩트 아님
- 모든 margin(M, ΔM) = 라벨러 독립
- **Run 13 "저계수 rescue 없음" null = SAFE** (라벨 bias는 null 강화만; full-state는 같은 라벨러가 +10.33pp로 잡음 → 숨은 rescue였다면 잡혔을 것; blind Codex 예측 일치)

**재계산 필요하지만 어느 것도 메커니즘을 되살리지 않음:** Run 13 절대 harm 수치(decoding_failure=0.0%는 구조적 오류) + 이미 대체된 Run 11 "52% rescue" → 저장된 generation 재라벨(무료).

**두 가지 estimand/label 뉘앙스 (사용자 요청대로 명시):**

1. **Run 4 P1 null은 intent-to-treat다.** audio가 공격을 "회피"하는 주 경로가 degrading이라, per-attempt(decoding-failure 제외)로는 격차가 glm **+7pp** / laguna **+12.5pp**로 커진다. 단 이는 treatment로 유발된 결과에 조건화한 값이라 **인과적 P1을 복원하지 않는다.** 정직한 문장 = "깨끗한 harmful-**compliance** 격차는 없다; audio 초과분 = degradation + over-refusal + mode-shift."
2. **공식적으로 재검토를 요청하는 유일한 지점 = emotion/paralinguistic null(Run 8).** 미검증 강도의 CosyVoice2 자극에서 나왔고, Emotion-Variations(2510.16893)는 emotion→unsafe가 **강도에 비단조**라고 보고 → 단일점 null로는 반증 불가. **classifier-검증 강도 dose-ladder로 싸게 재검**할 가치는 있으나, 기대 payoff는 낮다(StyleBreak: Qwen-Omni 0→9.1%로 modest).

## 3. 원래 가설 최종 판정 (문헌 접지)

| | 판정 | 근거 |
|---|---|---|
| **P1** (audio > text) | **KILLED** (load-bearing 형태) | Run 4 T0 +2.7pp n.s., DiD −26.7pp; JALMBench(Qwen2-Audio 6.9→7.3), Alignment Curse(2602.02557: text-transferred ≥ audio-native) |
| **M1** (조작가능 audio refusal 축) | **SPLIT** — trained actuator만 생존(unremarkable) | 자연 mediator FALSE: readout 0.60/escape 0.484/rescue≈random. Arditi(2406.11717)=축은 modality-특이 아님; SARSteer(2510.17633)가 이미 소유 |
| **S1** (per-attack DiM 분산/수렴) | **DISSOLVED** (애초에 답 불가한 거짓 이분법) | 고차원에서 geometric 분산 ≠ 메커니즘 수; 고정 저계수 L18 반증(Run 13); operator는 underdetermined지만 operator 성공≠safety-specificity |

## 4. Codex ↔ Claude-reviewer 합의 (수렴 확인)

두 리뷰어가 독립적으로 다음에 합의:
- **점수: as-is 3/10** (unaudited construct + §0 사실상 포기[~6 direction 문서, T0 hard-gate가 STOP 후 강등 = garden-of-forking-paths] + single model/TTS/family + novelty crowding), **2-model×2-TTS×family 재현 + blinded relabel 시 5–6.**
- **다음 스텝 순서: 저장 데이터 재라벨이 먼저** (새 렌더보다 strictly superior). primary contrast = **coherent intelligibility-matched control**(adversarial 지적), `pv_standard−pv_locked`는 phase-특이 **decomposition**으로 유지(Codex 지적) — 둘 다 필요.
- **White space: 얇고 negative.** HARC(2607.00572, text)·ReGap(2605.18104)·GACL(2606.05161)·MTAM(2509.24215)·AIA(2605.18168)가 대부분 소유. 미소유 = "content-preserving acoustic 교란이 **audited actionable harm 증가 없이** 큰 조작가능 refusal-geometry/template 실패를 만든다 + paired generative causal-validity audit가 genuine safety 메커니즘을 proxy/mode repair와 구분" — **재현 후에만 방어 가능**, 아니면 crowded negative replication → 피벗.

## 5. 확신 회복 경로 (ICLR 초석) — 사전등록 kill 기준

### Step 0 — 저장 출력 freeze & 재라벨 (linchpin, ~무료, GPU 불필요)
새 라벨을 보기 **전에** taxonomy/control/estimand/threshold를 동결. Run 10(246) + Run 13(300) 모든 paper-critical generation을 **조건/개입 blind 2-rater + 조정**으로 5-way 재라벨 {actionable-harmful / echo-transcription / safe-refusal / benign-relevant / decoding-failure}, **κ≥0.70**(1회 calibration 후).
- **KILL(safety-bypass thesis):** actionable이 outcome-blind coherent-matched control 대비 ≥+5pp AND clustered 95% CI가 0 배제 AND 잃은 refusal의 ≥50%가 actionable로 전환 — 아니면 死.
- **KEEP(measurement thesis):** template-loss ≥10pp AND actionable ≤+3pp(CI가 5pp 배제) AND refusal-loss의 ≥75%가 echo/description/benign/decoding.
- 둘 다 아니거나 κ<0.70 → **phase-vocoder 방향 전면 중단.**

### Step 1 — 표적 2-model × 2-TTS 재현 (Step 0가 measurement thesis 지지할 때만)
Qwen2-Audio + 2nd arch; CosyVoice2 + 2nd TTS; arms = harmful + capability-matched benign + neutral-transcription + 일반 instruction-following; conditions = clean / pv_standard / **frozen coherent intelligibility-matched control(primary)**; `pv_standard−pv_locked`=phase decomposition, `clean−attack`=total. **검정력은 Step 0 효과크기로**(임의의 400 금지).
- **KILL:** proxy-behavior dissociation이 ≥3/4 model×TTS cell에 없거나, pooled actionable-harm이 <5pp equivalent 아니거나, non-actionable shift가 erosion의 ≥75%를 설명 못하면 ICLR measurement thesis 폐기. **한 모델이 genuine actionable compliance를 보이면 → 그 model/regime로 피벗**(평균으로 뭉개지 말 것).

### Step 2 — 봉인된 causal-abstraction 테스트 (untouched split, frozen intervention, layer/rank/dose search 금지)
frozen safety 좌표/subspace + norm-matched sham + full-state clean restore + harmful/benign/neutral-transcription arms.
- **원하는 결과:** 좌표는 margin/template를 움직이나 actionable harm는 아님; full-state는 transcription/task-mode를 generic하게 repair.
- **KILL:** 좌표가 유효 proxy 효과(ΔM≥0.05, CI 0 배제) + actionable-harm <5pp equivalent + full-state가 safety·non-safety task-mode 둘 다 개선 — 아니면 메커니즘 기여 폐기. **좌표/operator가 선택적으로 genuine safety를 복원하면 measurement thesis가 틀린 것이고 새 메커니즘 프로젝트 정당화.**
- **Step 1이 real safety phenotype를 확립하기 전 temporal operator 탐색 금지.**

## 6. Bottom line

첫 가정은 load-bearing 형태로 틀렸다: Qwen2-Audio는 여기서 깨끗한 audio-specific harmful-compliance 취약성을 보이지 않았고, estimand에 따라 심각도가 달라지는 **audio-conditioned response instability**만 보였다. 진짜 ICLR 경로는 **재현된 measurement/causal-proxy audit로서만** 존재한다. Step 0(무료) 재라벨이 그 결과를 내지 못하면 phase-vocoder/Qwen2-Audio 코너 파기를 멈추고 **adversarial/AIA regime** 또는 **genuine actionable compliance가 비자명한 모델**로 옮겨라.

---

## 부록: 이번 세션에서 web-검증한 참고문헌

- **JALMBench** 2505.17568 (ICLR'26): Qwen2-Audio text 6.9% / audio 7.3% ASR; text-transferred(37.5%)<text-input(49.7%); audio-originated 72.9%, AdvWave 97.3%.
- **Acoustic Interference / AIA** 2605.18168 (ICML'26): benign-content+ALS universal trigger로 malicious text 우회; refusal-vector late-layer negative drift + bidirectional patching; 10 LALM incl Qwen2.5-Omni.
- **Alignment Curse** 2602.02557: text-transferred audio jailbreak ≥ audio-native (neutral TTS regime).
- **HARC** 2607.00572: harmfulness↔refusal 방향 dissociation + coupling 방어 — **text LLM**, 5 families.
- **ReGap** 2605.18104: multimodal drift가 refusal separability 압축 + activation-correction 복원 + self-rectification harmfulness 신호.
- **Beyond Text Following / GACL** 2606.05161: same-audio counterfactual(64.1% sign flip) + answer-position 국소화(ρ=0.93) + logit-correction repair, **5 ALM**.
- **MTAM** 2509.24215: acoustic metamorphic-invariance 측정.
- **Emotion-Variations** 2510.16893: emotion→unsafe가 강도에 비단조.
- **StyleBreak**(AAAI): paralinguistic 공격 — Qwen-Omni 0→9.1%, 최대 +26%(특정 모델/스타일).
- **LISTEN**(2026): LALM이 lexical content 대비 acoustic emotion을 underuse.
- **SARSteer** 2510.17633: text-derived refusal steering이 Qwen2-Audio에 전이; audio-contrast는 분포 gap으로 신뢰불가.
- **Arditi** 2406.11717: 단일 refusal 방향(modality-특이 아님).
