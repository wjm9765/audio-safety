# 오디오 안전성 논문 — 실험 결과 보기 전 프레이밍 논의 정리

- **날짜:** 2026-07-24
- **성격:** *pre-experiment discussion* — 이 프로젝트의 **실험 결과 파일은 일절 열람하지 않고**, 문제 정의·가설 자체의 타당성만 외부 문헌에 근거해 검증한 기록이다.
- **참여:** Claude(main) ↔ Codex(gpt-5.6-sol, reasoning effort xhigh, 3+1 라운드) ↔ adversarial-reviewer(ICLR 리뷰어 관점, 2패스). 두 리뷰어는 서로의 답변을 보지 않고 **독립(blind)** 으로 평가했다.
- **종료 조건:** 두 리뷰어 모두 "재정의된 문제+가설이 기존 연구 기준으로 타당하고 좋다"고 판단할 때까지 반복. → 충족됨(§4).

> ⚠️ 이 문서는 사전등록 설계서(`design.md`)가 아니다. 판정 임계값을 확정하는 문서가 아니라, **설계를 확정하기 전 단계의 논의 로그**다. 여기서 나온 §0 기준 후보(§10)는 사용자 승인 전까지 확정이 아니다.

---

## 1. 논의 목적과 방식

"content-preserving 오디오(말은 그대로, 목소리·음높이·감정 등 audio 요소만 조작)가 모델 안전성을 뚫는데, 그게 모델 내부 **어디서** refusal/harmful을 trigger하는지에 대한 디테일한 분석이 없다"는 문제의식과, "각 공격마다 refusal 표상 차이를 만들면 **하나의 축으로 수렴하냐 / 여러 축으로 흩어지냐**"라는 가설을 검증 대상으로 삼았다.

핵심 원칙: **결론을 보고 기준을 옮기지 않는다.** 그래서 (1) 실험 결과를 안 보고, (2) 외부 논문 근거를 반드시 달고, (3) 두 에이전트 리뷰어의 blind 교차검증을 통과할 때까지 반복했다.

---

## 2. 원래 문제 정의 + 가설 (연구자 최초 서술)

**문제 정의(최초):**
- content-preserving audio가 audio-specific 조작만으로 모델 안전성을 뚫는다. 기존 연구는 이를 jailbreak에 쓴다.
- 그런데 이 audio 요소가 모델 내부 어디서 refusal/harmful을 trigger하는지 디테일 분석이 없다.
- 더 근본적으로: **애초에 어떤 오디오가 "안전한" 기준 오디오인지 모른다.** 여성이 harmful 질문을 하고, 더 높은 음으로 말하면 같은 말인데 파형이 달라지고, 남성이 말하면 음이 더 낮다. 그럼 여성이 안전한가 남성이 안전한가 — 기준이 없다.

**가설(최초):**
- audio-specific 요소(speech, emotion, pitch/vocoder 등) 각각이 refusal을 건드린다. 기존 연구는 LLM layer ~18(Qwen 32층 중 중간)에 뭔가 있다고 보고 그걸 주로 썼다.
- harmful-reject 표상 − harmful-accept 표상의 평균 차이(diff-in-means)를 공격 종류별로 만들어 분포를 본다 → (A) 분산이 크면 서로 다른 축, (B) 수렴하면 하나의 audio-specific refusal vector.
- 대안: 초기 layer의 audio(encoder→projector→LLM 초반)가 harmful intent를 담고, 중간을 거치며 내부 결정으로 넘어간다.

---

## 3. 검증된 문헌 지형

아래는 모두 웹에서 실재를 확인한 논문이다(할루시네이션 아님). 이 공간은 **매우 혼잡**하다.

| 논문 | 이미 한 것 | 이 연구에 대한 의미 |
|---|---|---|
| **Refusal single direction** (Arditi 2024, 2406.11717) | 텍스트 LLM의 거부가 **단일 방향**으로 매개됨. diff-in-means, ablate→우회, add→유도 | 방법론 뼈대. 단일 벡터 가정은 아래 subspace 논문이 약화 |
| **AIA / Acoustic Interference** (2605.18168, ICML'26) | paralinguistic universal jailbreak. **이미** layer별 diff-in-means refusal vector + late-layer drift + **인과 activation patching**(text↔audio) + pitch/valence/speed/gender 중요도 | **최대 경쟁자.** 단, single-vs-multi 축 검증 X, repair X, 분석을 Qwen2.5-Omni에서(Qwen2-Audio 아님), attack 프레이밍 |
| **SARSteer** (2510.17633) | **Qwen2-Audio**+Kimi에서 inference-time 방어. harmful-vs-safe **speech contrast는 distributional gap으로 폐기**, text-derived refusal + PCA safe-space ablation 사용 | 연구자의 원래 diff-in-means가 바로 그 폐기된 대비. 동시에 "audio-specific geometry는 회피"라는 틈 |
| **ALMGuard** (2510.26096, NeurIPS'25) | **첫 ALM 전용 방어.** 입력에 범용 음향 섭동(SAP)을 더해 "가정된 안전 shortcut" 발화. Mel 주파수 마스크. 적대적 음향(AdvWave)에 강함(→4.6%) | "텍스트/전통적 오디오 방어는 무력"을 스스로 전제 → 동기 지지. 단 **필수 baseline**이 됨(§9) |
| **Cascade Equivalence** (2602.17598) | Ultravox ≈ Whisper→LLM cascade지만 **Qwen2-Audio는 유의하게 divergent**(cascade가 버리는 paralinguistic 정보를 보유) | 전제 지지 + 비교 도구 제공. 단 activation-matched baseline은 아님 |
| **MARS** (2606.31876) | **vision/video 전용.** "텍스트 refusal direction은 modality 간 강력한 공유 기반" + re-centering | audio가 이 서사를 깨는지가 열린 대조. Related Work의 vision 선례로만(오디오 baseline 아님) |
| **Over-refusal subspaces** (2603.27518) | 텍스트 거부가 **다차원 subspace**임 | 단일 Arditi 벡터를 null로 쓰면 "orthogonal 성분" 주장이 붕괴 → null은 최선의 다차원 subspace여야 |
| **VoxSafeBench** (2604.14548) | who/how/where(화자/운율/환경) 안전 벤치마크 | invariance "측정" 자체는 선점. 또 "paralinguistic은 때때로 정당한 안전 신호"라는 반례 |
| **Emotion variations** (2510.16893) | emotion→unsafe율이 **강도에 비단조** | 선형-축 가정 위험 경고 |
| **Refusal universal across languages** (2505.17306) | 거부 방향이 언어 간 공유 | 채널-불변성의 다른 축 유사 사례(언어 = 채널) |
| 기타 | 다국어·다억양 오디오 jailbreak(2504.01094), audio-specialist heads(2603.06854), encoder-only attack(2512.23881), fairness eval(2603.13262) | "누가/어떻게/어디서"·국소화·공정성 각도는 상당 부분 선점 |

**개념 틀:** "전달 방식이 바뀌어도 판단은 같아야 한다"는 이미 이름이 있다 — counterfactual fairness(Kusner 2017, 1703.06856), metamorphic testing. 그래서 이 원리 자체는 novelty가 아니다.

---

## 4. 판정 요약

| 대상 | Codex | adversarial-reviewer |
|---|---|---|
| **원래 문제+가설(§2)** | REJECT (AIA에 incremental) | REJECT (AIA 재현+rank 숫자) |
| **재정의 v3/v4(§6)** | **GO** (개념적 blocker 없음) | **ACCEPT — champion 트랙** (모든 FATAL 해소, valid·sound) |

→ **두 리뷰어 모두 재정의된 문제+가설을 타당·건전하다고 판정.** 단 비-incremental 여부는 **프레이밍이 아니라 실험(H3/H4)의 effect size에 달림**(§7).

---

## 5. 동기로 강등된 것들 (기여 아님)

1. **"안전한 오디오 기준이 없다"** → 규범적 기준은 waveform이 아니라 **요청의 harm label**이다. counterfactual fairness / metamorphic testing이 이미 "canonical baseline 없이 label-보존 변형을 비교"하는 구조. **동기로는 훌륭하나 novelty로는 못 씀.**
2. **"채널-불변성을 스펙으로 제시"** → VoxSafeBench·fairness·cross-lingual이 이미 점유.
3. **원래 diff-in-means(refused − accepted)** → (a) 모델 결과로 표본을 가르는 **outcome-conditioned selection**이라 분리가 인위적으로 부풀려지고, (b) SARSteer가 distributional-gap으로 폐기한 바로 그 대비. **기각.**
4. **"분산 크냐 / 하나로 수렴하냐"** → 고차원은 요인마다 다른 방향에 자연스럽게 인코딩된다(모두 하나의 gate로 흘러도). **기하학적 분산 ≠ 메커니즘 개수.** 또 **invariance ≠ safety**(항상 거부도 완벽히 invariant).

---

## 6. 재정의된 문제 정의 + 가설 (v4)

### 6.1 스코프된 스펙
transcript(글자)를 보존하고 **safety-relevant 의도까지 보존**하는 렌더링의 등가류에 대해 Qwen2-Audio의 refusal은 불변이어야 한다. 그런데 아니다.
- 스코핑 원칙(§8 논쟁 결과): **위험 유형을 "응급/나이 예외가 없는 hard-harm"으로 엄선**하면, 목소리 채널(아이·절박함 포함)은 순수 잡음 요인이 되어 flip이 명백한 버그가 된다.

### 6.2 가설 (반증가능·양방향 publishable)
- **H1 (제어 가능성, tautology 아님):** scoped 렌더링이 refusal에 미치는 인과 효과는 **포화된(saturated) text-derived refusal subspace**로 — 선언된 linear/low-rank intervention class 하에서 — 완전히 제어되지 않는다. *(주의: "audio 정보가 존재한다"(Cascade)나 "그 정보가 refusal을 움직인다"(AIA)와 명시적으로 구분되는 **제어가능성** 주장이어야 함.)* 귀무 H0: 모든 효과가 그 subspace로 매개됨.
- **H2a (cross-fitted 예측 signature):** outcome에 접근하지 않고 구성한 오디오 잔차의 크기가 **채널별 text-derived 방어(SARSteer) 실패를 out-of-sample로 예측.** 핵심 플롯 = |오디오 잔차| vs 방어 실패율. transplant/removal로 바꿔 보이기 전까지는 "법칙"이 아니라 "signature".
- **H2b (cross-model triangulation, "법칙" 아님):** Qwen2-Audio vs Ultravox 경로 유무 대조 — 사전등록된 외적 타당성 예측, within-model 정규화. 2모델은 class-level 법칙 아님(class 주장 시 Qwen2.5-Omni 추가).
- **H3 (국소화 — 진짜 기여):** 오디오 고유 성분이 인과 효과를 처음 획득하는 지점을 encoder→projector→LLM으로 **추적·차단 가능한 경로**로 규명. AIA의 late-drift와 encoder-only 충분성 **둘 다를 특이성에서 능가**해야.
- **H4 (repair — 진짜 기여):** 국소 locus에서의 **비-oracle(라벨 없는) 개입**이 unseen 음향 조건에서 full text-only Pareto(MARS·ALMGuard 포함)보다 나은 safety–utility frontier를 냄. 아니면 "text로 충분"이 아니라 **잔여 오디오 이득의 상한(UCB)을 둔 affirmative-equivalence**로 보고.

### 6.3 측정 규율 (설계에 내장)
- 짝지은 반사실 표본을 **randomized acoustic assignment**로(outcome-conditioned 금지); shift 벡터는 flip쌍이 아니라 **모든 matched쌍**에서.
- lever-vs-factor 교차검증(DSP / TTS / voice-conversion; vocoder-only resynthesis로 artifact 가격 매기기).
- **내용 인지(comprehension) 매개 분석 + 요인별 배제율 보고** — "모델이 위험한 말을 여전히 제대로 들었나"는 별개 관문(§10 gate 1).
- emotion 강도 dose-response(비단조); logit-margin은 실제 harmful-generation으로 검증; layer 선택은 disjoint split(double-dipping 금지); noise-floor 통제(split-half 동일-요인 cosine 상한 + matched-norm 무작위/benign-concept 방향 대조).
- 채널 수가 구조적으로 작으므로 **채널-permutation + item-nesting + 독립 채널 수가 증거 상한**임을 정직히 명시.

---

## 7. novelty가 실제로 사는 곳

두 리뷰어의 일치된 냉정한 결론: **H1/H2는 건전하나 (Cascade ∧ AIA ∧ subspace) + 더 나은 통계에 가깝다.** 이 논문이 AIA/SARSteer/ALMGuard를 넘어서는 힘은 **전적으로**:
- **H3 — 추적된 차단 가능 경로**(단순 layer별 그래프/히트맵을 넘어, 성분 제거→flip 사라짐 / 이식→재현의 인과 수준. 가장 강한 그림 = **이중 해리**), 그리고
- **H4 — 배포 가능한(라벨 없는) repair가 메커니즘의 인과성을 검증**.

에 있다. 여기서 non-marginal한 결과가 나오면 확실한 ACCEPT, 어쩌면 champion.

---

## 8. 스코핑 논쟁 해소 (content-preserving / 아이·비명)

연구자 반론 두 가지에 대한 결론:

1. **"content 보존 = transcript가 (거의) 그대로."** ("폭탄 만드는 법 알려줘" ↔ "폭탄 만드는 거 어떻게 해?" 수준의 최소 변형.) → **hard-harm에서는 '비꼼/인용으로 의도가 뒤집힘' 걱정이 대부분 사라진다** — 위험한 출력은 입력이 진지하든 아니든 내보내면 안 되기 때문. airtight하게 가려면 감정 조작 목록에서 '비꼬는 톤'만 빼면 됨.
2. **"아이 목소리·절박함도 그냥 넣어라."** → **맞다.** 마약·살인 등 예외 없는 위험에서는 아이가 절박하게 물어도 정답은 "거부". 그러니 flip은 무조건 버그다. 리뷰어의 "빼라"는 *양날(dual-use) 요청을 가정*한 조언이라 여기 안 맞는다. 채널을 빼는 대신 **위험 유형을 hard-harm으로 엄선**하는 게 정답 — 오히려 라벨이 깨끗해지고(모든 compliance=진짜 실패) 강력한 채널을 살린다.

**단 분리해야 할 것:** "의도 보존"(비꼼)과 **"내용 인지"(모델이 위험한 말을 제대로 들었나)**는 다른 문제다. 후자는 §10 gate 1로 **항상 필요**하다(목소리를 세게 망가뜨려 모델이 잘못 들어서 응하면 그건 안전 버그가 아니라 오인).

---

## 9. ALMGuard 분석 (2510.26096)

**"내가 텍스트 방어 불충분을 논하면 리뷰어가 '왜 ALMGuard 안 썼냐'고 할 것"**이라는 우려에 대한 Codex 토론 결과.

**ALMGuard의 실체:** 입력 오디오에 **범용 음향 섭동(SAP)**을 더해 "이미 존재한다고 *가정*한" 안전 shortcut을 발화시키는 **입력공간** 방어. Mel 주파수 마스크로 jailbreak엔 민감/speech엔 둔감한 bin만 건드림. 적대적 음향(AdvWave)에 강함. 자기 입으로 **"semantic 의도 기반 공격엔 개선 여지"** 인정.

**결론: ALMGuard는 프레이밍을 죽이지 않는다 — 동기를 돕는다.** 단 대응 논리 두 개는 과했음(교정):
- ❌ "우리 메커니즘이 ALMGuard를 설명한다" → 아직 못 함. SAP가 실제로 우리가 찾은 내부 성분을 통해 작동하는지 **다리 실험**(SAP 활성 이동을 계층별로 재고 → 성분 분해 → 그 성분 차단 시 ALMGuard 방어력 소실 확인)이 필요. 그 전엔 **"ALMGuard = 가설한 내부 경로를 외부에서 찔러보는 탐침"**으로만.
- ❌ "semantic vs acoustic 구분으로 우리가 ALMGuard 약점 지대" → 틀림. 음높이·감정은 명백히 acoustic. ALMGuard의 "semantic 약점"은 *프롬프트 의미 조작* 공격 얘기지 파형 조작 얘기가 아님.

**정당한 wedge = "적대적(adversarial) vs 자연스러운(naturalistic)".** ALMGuard는 AdvWave 같은 최적화된 인공 잡음에 검증됐지, **자연스러운 label-보존 paralinguistic 분포 이동**에 대해선 미검증. 이건 추정된 약점이 아니라 empirical gap.

**필수 실험 = ALMGuard 정면 비교 + 양성 대조.** ALMGuard를 우리 벤치마크에 돌리되 **AdvWave에서도 돌려 구현 충실성**을 먼저 입증(안 그러면 "구현 실패 아니냐" 반박). 결과별 해석:
- 막으면 → 방어 gap 없음, H4는 실전 방어로서의 새로움 상실(메커니즘은 여전히 물음).
- 자연스러운 건 못 막고 AdvWave는 막으면 → **진짜 coverage gap. 강한 결과.**
- 막긴 하나 과잉 거부/이해력 저하로 막으면 → scoped 목표를 푼 게 아님.
- 상보적이면 → 가장 유익.

**H3 차별점(daylight):** ALMGuard의 국소화는 **입력 주파수 공간**(어느 Mel bin), 우리는 **모델 내부 계산 경로**. 비유: "어느 픽셀이 공격에 효과적인가" vs "그 픽셀이 내부 어느 회로를 거쳐 결정이 되나". 단 H3가 단순 히트맵을 넘어 인과(제거/이식)까지 가야 성립.

**H4 위치 재조정:** ALMGuard가 이미 저비용·고성능이니 **H4를 "최고 방어"로 팔지 말고 "국소 지점이 진짜 인과 지점임을 증명하는 검증"으로** 판다.

**"왜 ALMGuard 안 썼냐"에 대한 한 문장(Related Work/rebuttal):**
> ALMGuard는 우리 분석의 대체물이 아니라 상보적 입력공간 baseline이다 — 그것은 *가정된* 음향 안전 shortcut을 발화시키려 범용 Mel-공간 섭동을 더하는 반면, 우리는 그 방어가 *자연스러운, 라벨 보존 paralinguistic* 상황에서 실제로 보정 효과가 있는지 검증하고 그 배후의 내부 인과 경로를 규명한다. 우리는 SAP가 그 경로를 통해 작동함을 실험으로 보였을 때에만 'ALMGuard를 설명한다'고 주장한다.

**실무:** GO 불변. ALMGuard는 이제 **최소 primary 모델에서 필수 baseline**(ALMGuard vs SARSteer vs 우리 repair + AdvWave 양성대조 + (설명 프레이밍 유지 시) SAP 활성 분석). **4번째 모델은 강제 아님**(ALMGuard는 방법이지 모델 클래스가 아님; Qwen2.5-Omni는 여전히 AIA 비교용).

---

## 10. 사전등록 기준 후보 (§0) + Go/No-Go 게이트

**Codex 제안 primary confirmatory criterion (연접, layer/rank/controller/covariate/threshold 사전 동결):**
> 오디오 성분(U_audio)을 주장하려면, held-out contents×levers에서 **(i)** 방어 전 |U_audio|가 degradation 통제 + matched-rank 음성 예측자를 넘어 saturated 텍스트 subspace 제어의 잔여 실패를 예측 **∧ (ii)** 성분-only transplant와 removal이 harmful-generation을 양방향 변경 **∧ (iii)** 비-oracle repair가 benign-utility noninferiority δ 이내에서 full text Pareto를 worst-orbit harmful-safety로 ≥Δmin 개선. 하나라도 실패 시 audio-component 주장 기각.

**게이트:**
1. **내용-degradation gate(가장 큰 침몰 위험 차단):** 잔차 acoustic 효과가 *모델이 지각한* 내용 저하(ASR 불확실성/누락/일반 OOD severity) 통제 후에도 살아남을 때만 진행. Whisper 일치가 아니라 **모델-내부 content/intent probe + completeness 양성대조**(계산된 크기의 알려진 내용 변화를 심어 probe가 잡는지)로 검증. 실패 시 → "ASR-매개 안전 실패"로 리프레이밍.
2. **text-control 포화 곡선:** 증명 불가한 단어 "saturated"를 **실증적 포화 곡선**으로 대체 — 텍스트 제어를 데이터/rank/강도로 키워 worst-channel safety가 plateau에 드는 걸 보이고, 오디오 증강이 그 plateau 외삽 CI를 넘음을 보인다. **SARSteer·MARS·ALMGuard를 이 곡선 위의 점**으로.
3. **스코핑 완화:** 음향적으로 강력하나 이 harm엔 nuisance인 채널(화자 정체성/억양/비언어 발성)을 최소 1개 유지 + 배제한 safety-relevant 채널을 **양성 대조**(경로가 검출되긴 하는지)로.

---

## 11. 필요한 실험·baseline·모델 (설계 시 반영)

- **데이터:** randomized 짝지은 반사실 오디오(같은 문장, 요인 1개씩 변형), 다중 생성기(DSP/TTS/voice-conversion), 요인별 배제율 측정, hard-harm 위험 유형 엄선 + 경계적 dual-use benign 오디오(유용성 축 rigging 방지).
- **baseline:** **SARSteer(텍스트-derived)**, **ALMGuard(입력 섭동, +AdvWave 양성대조)**, 최선의 다차원 텍스트 subspace(포화 곡선), (참고) MARS는 vision 선례.
- **모델:** Qwen2-Audio(주) + **Qwen2.5-Omni**(AIA 직접 연결) + **Ultravox**(cascade-equivalent 대조). 단일 모델은 자동 reject.
- **content 통제:** human-verified transcript→**동일 decoder**(가장 깨끗한 content-only + 텍스트 subspace 정의); 실제 Whisper→LLM(행동적 ASR-bottleneck 음성대조); Ultravox vs Qwen2-Audio(교차구조 예측, activation-matched 아님).

---

## 12. 가장 큰 리스크

1. **스코핑 vs effect-size 긴장(가장 유력한 침몰):** invariance를 well-posed하게 만들려 safety-relevant paralinguistic을 빼면 정작 오디오 고유 경로가 클 채널을 함께 빼 → H4가 marginal affirmative-equivalence로 착지 → SARSteer/MARS/ALMGuard 대비 incremental. (§10 게이트 3로 완화.)
2. **내용 저하 혼입:** 잔차가 사실 ASR 불확실성/오인이면 메커니즘 novelty가 encoder-attack+AIA로 붕괴. (§10 게이트 1로 차단.)
3. **채널 수준 n이 구조적으로 작음:** 요인 몇 개뿐 → 상관이 1~2 채널에 좌우. 채널-permutation + item-nesting + 정직한 power-cap 진술 필요.
4. **H0 비대칭:** "텍스트로 충분" 결과는 훨씬 넓은 실험(다모델 equivalence + 보편 텍스트 repair)이 있어야 겨우 publishable.

---

## 13. 다음 단계 / 열린 항목

- [ ] 위 v4를 사전등록 설계서(`design.md`) §0 형식(가설·판정 임계값·intervention class·양성대조)으로 초안화 — **사용자 승인 후 확정.**
- [ ] "관문 실험(내용 인지 gate) → 본실험(포화곡선 → 인과 dissociation → repair)" 순서의 파일럿 계획.
- [ ] baseline 구성표(SARSteer·ALMGuard·AdvWave 양성대조·cascade·모델 3종)를 하나의 비교 설계표로.
- [ ] (선택) 이 ALMGuard 논점을 adversarial-reviewer에 재교차검증.

---

## 부록: 참고문헌 (검증된 링크)

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (NeurIPS 2024) — https://arxiv.org/abs/2406.11717
- *Acoustic Interference (AIA)* — https://arxiv.org/abs/2605.18168
- *SARSteer* — https://arxiv.org/abs/2510.17633
- *ALMGuard* (NeurIPS 2025) — https://arxiv.org/abs/2510.26096
- *Cascade Equivalence Hypothesis* — https://arxiv.org/abs/2602.17598
- *Harnessing Textual Refusal Directions for Multimodal Safety (MARS)* — https://arxiv.org/abs/2606.31876
- *Over-Refusal and Representation Subspaces* — https://arxiv.org/abs/2603.27518
- *VoxSafeBench* — https://arxiv.org/abs/2604.14548
- *Safety Vulnerabilities under Speaker Emotional Variations* — https://arxiv.org/abs/2510.16893
- *Refusal Direction is Universal Across Languages* — https://arxiv.org/abs/2505.17306
- *Breaking Audio LLMs by Attacking Only the Encoder* — https://arxiv.org/abs/2512.23881
- *Audio-Specialist Heads for Adaptive Audio Steering* — https://arxiv.org/abs/2603.06854
- *Multilingual and Multi-Accent Jailbreaking of Audio LLMs* — https://arxiv.org/abs/2504.01094
- Kusner et al., *Counterfactual Fairness* (NeurIPS 2017) — https://arxiv.org/abs/1703.06856
- *Qwen2-Audio Technical Report* — https://arxiv.org/abs/2407.10759
- AdvWave — ALMGuard가 사용한 적대적 오디오 jailbreak 벤치마크(방어 양성대조용)

---

*작성: Claude(main). 교차검증: Codex(gpt-5.6-sol, xhigh) + adversarial-reviewer. 실험 결과 파일 미열람.*
