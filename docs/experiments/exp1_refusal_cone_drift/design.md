# 실험 1: Audio-Induced Refusal-Cone Drift Probe (Qwen2-Audio)

> **한 줄 목표:** "서로 다른 audio 공격 family가 refusal subspace의 *서로 다른 축*을 흔드는가(H1), 아니면 전부 *하나의 축*으로 collapse하는가(H0)"를 며칠 안에 판정한다. 학습 없음. 방어 method 없음. baseline 재현 없음. 오직 이 그림 하나.

---

## 0. 가설과 판정 규칙 (먼저 못박기)

- **H0 (논문 죽음 / Safety Geometry Collapse 재탕):** 모든 family의 drift가 같은 단일 refusal 축 `b1`과 정렬된다. family별 projection 프로파일이 같은 방향의 스칼라배일 뿐(pairwise cosine ≈ 1), 크기만 다르다.
- **H1 (논문 생존):** family마다 dominant 축이 다르다. drift 프로파일의 pairwise cosine이 낮고, "어느 축을 가장 흔드는지"가 family별로 갈린다.

**판정 metric = family별 drift 프로파일 벡터들의 mean pairwise cosine + permutation p-value.**

| 결과 | 판정 | 다음 행동 |
|---|---|---|
| mean pairwise cosine **< 0.6**, dominant 축이 ≥2 family에서 다름, permutation **p < 0.05**, 축이 causal ablation 통과 | **GO (H1)** | baseline 재현 → multi-direction risk-adaptive steering method로 진행 |
| mean pairwise cosine **≳ 0.85** (단일 축 collapse) | **NO-GO (H0)** | 메커니즘 경쟁 포기. cost/utility-aware inference-time 방어(ALMGuard 대비 over-refusal·latency 우위)로 즉시 피벗 |
| 그 사이(0.6–0.85) | **AMBIGUOUS** | 카테고리 축 추가 + n 증가 + comprehension 필터 강화 후 재측정 |

이 표를 실험 시작 전에 박아두는 이유: 결과를 보고 사후에 기준을 옮기는 self-deception을 막기 위해서다.

---

## 1. 설계 핵심: paired drift design (confound 제거)

순진하게 "cone 축 = harm 카테고리, family별로 projection 비교"를 하면, family 간 **content가 고정**되므로 content 축에는 다 비슷하게 실려 H0로 인위 편향된다. 그래서 측정 대상을 **modality manipulation이 유발한 drift 벡터**로 바꾼다.

- **Anchor:** 같은 harmful 내용 `c`의 **clean text** 표현 `h_text(c)` — refusal geometry가 깨끗하게 사는 곳.
- **각 family `f`:** 동일 내용 `c`를 family `f`로 렌더링한 audio 입력의 표현 `h_f(c)`.
- **Drift 벡터(측정량):** `d_f(c) = h_f(c) − h_text(c)` — content를 짝(pair)으로 묶어 상쇄. 즉 family는 within-item factor.
- **질문:** `d_f(c)`를 refusal cone 축 `[b1..bk]`에 projection했을 때, family별 프로파일이 다른가?

이 구조의 장점: content·길이·주제 confound가 pairing으로 자동 제거되고, 남는 변량은 순수하게 "modality가 refusal 표현을 어떻게 비트는가"다. 통계도 repeated-measures라 검정력이 높다.

**요인 구조**
- 독립변수(within-item): attack family `f` ∈ {plain, nonspeech, style, perturbed}
- 매칭 단위: harmful content `c` (family 간 동일 내용 공유)
- 종속변수: cone 좌표계에서의 drift 프로파일 `proj(d_f(c)) ∈ R^k`

---

## 2. 데이터

### 2.1 Refusal cone 정의용 (text only)
- **Harmful text `H`:** AdvBench / HarmBench / SORRY-Bench 중 카테고리 라벨이 있는 것. 의미적으로 구분되는 harm 카테고리 6개 선정 권장: 무기·폭력 / 사기·금융 / 멀웨어·해킹 / 프라이버시·doxxing / 혐오·괴롭힘 / 불법물질. (카테고리별 약 256~512쌍)
- **Benign/borderline text `B`:** XSTest(over-refusal borderline-safe 전용 벤치) + Alpaca/just-eval 일반 benign. borderline을 꼭 넣어야 "harmful topic 탐지"가 아니라 "refusal" 축을 잡는다.
- 용도: 이 둘로 cone basis와 primary 축 `b1`을 만든다(§4). **학습 없이 diff-in-means/PCA.**

### 2.2 Drift probe용 (paired content × family)
- **Base 내용 `C`:** §2.1의 harmful 중 **150개 내용** 고정 선택(카테고리 균형). 이 150개를 모든 family로 렌더링 → paired.
- **Family 렌더링**
  1. **plain:** 내용을 표준 TTS로 그대로 읽은 음성. (AJailBench-Base 류; ALMGuard가 푼 `WeifeiJin/AdvBench-Audio`를 재사용해도 됨)
  2. **nonspeech:** plain 음성 앞/뒤/믹스로 non-speech 오디오(환경음 등) 추가. (AIAH의 "non-speech 추가만으로 표현이 이동" 가설 검증용)
  3. **style:** 동일 내용을, emotion/accent/age/gender/speaking-rate를 바꾼 음성으로 렌더링. (StyleBreak 류) 스피커 idiosyncrasy를 축이 잡지 않도록 family 내부에서 **≥3개 voice/style seed**로 분산.
  4. **perturbed:** plain 음성에 AJailBench의 Audio Perturbation Toolkit(time/frequency/amplitude 왜곡) 적용. **AdvWave식 풀 gradient 최적화는 비싸니 첫 실험에선 제외**(GO 이후로 미룸).
- **Anchor:** 같은 150개 내용의 clean text 입력.
- **Benign control(별도):** XSTest borderline 50~100개를 plain TTS로 — separation/over-refusal geometry 기준선.

### 2.3 표본 수 요약
- cone 축 추정: 카테고리당 256~512 text 쌍.
- drift probe: 150 내용 × 4 family ≈ 600 audio forward + 150 text anchor. comprehension 필터 후 family당 최소 ~100 유효 샘플 확보 목표.

---

## 3. 전처리 & confound 필터 (reviewer-proofing)

각 audio 샘플에 대해:
1. **Comprehension 필터(필수):** 모델/Whisper로 audio를 전사 → 원 내용과 의미 일치하는지 확인. 일치하지 않으면(특히 perturbed·style) 제거. *이걸 안 하면 "거부"가 안전성 refusal인지 단순 미인식인지 구분 불가 — 가장 흔한 reject 사유.*
2. **Refusal vs comprehension 분리:** 최종 분석은 "내용을 이해한(전사 성공) 샘플"에만 적용.
3. **Readout 위치 고정:** 모든 조건에서 동일 token 위치(생성 직전 마지막 입력 토큰)의 residual stream을 읽는다.
4. **Benign matched:** content 주제 분포를 harmful과 맞춰 "주제 탐지"로 새지 않게.
5. **Voice seed 분산:** style family는 다중 스피커로.

---

## 4. Refusal cone 구성 (학습 X) + causal 검증

### 4.1 Activation 추출
- 모델: `Qwen2-Audio-7B-Instruct`. 백본 LM = Qwen2-7B (decoder-only). **실제 layer 수·hidden_size는 `model.config.num_hidden_layers`, `model.config.hidden_size`로 확인**(대략 28 layers, hidden 3584이지만 코드에서 검증할 것).
- 각 decoder layer 출력(residual stream)의 마지막 입력 토큰 hidden state를 hook으로 수집. 모든 layer 저장.

### 4.2 Layer 선택 (sweep, 임의 선택 금지)
- text `H`/`B`로 layer별 diff-in-means separation(예: harmful/benign 평균 hidden의 거리, 또는 `b1` 방향 분리도)을 계산.
- separation이 최대인 layer를 **primary refusal layer `L*`**로. (refusal은 보통 중간 깊이 ~50–70%에서 강함.) 전체 sweep을 Fig 1로 보고.

### 4.3 Cone basis 만들기 (두 방법 모두, `L*`에서)
- **Method A (interpretable):** 카테고리별 diff-in-means `mean(H_cat) − mean(B)` 6개 → Gram–Schmidt 직교화 → `B_cone = [b1..bk]`.
- **Method B (cross-check):** 샘플별 차이 벡터 `{h_i^harm − mean(B)}`에 PCA → top-k 성분.
- 두 방법의 subspace가 대체로 일치하는지(principal angle) 확인 → robustness.
- `b1` = 가장 분리도 큰 단일 축(Arditi식). 나머지가 "추가 독립 축" 후보.

### 4.4 Causal ablation 검증 (이게 "임의 PCA 축" 반박을 막는다)
각 후보 축 `bj`에 대해, hold-out harmful **text** 프롬프트에서:
- **Ablation:** layer ≥ `L*`에서 `bj` 성분을 projection-out한 채 생성 → refusal rate 하락하면 그 축은 refusal을 매개(인과).
- **Addition:** 원래 응한 harmful 프롬프트에 `bj`를 더하면 refusal 증가하면 추가 확인.
- refusal 판정 = Arditi식 거부 문자열 매칭 + 일부 표본 LLM judge.
- **인과성 통과한 축만 cone에 유지.** (통과 못한 PCA 축은 버림.)

> 산출: Table — 축별 ablation 시 refusal-rate 하락폭. "이 k개는 진짜 refusal 방향"임을 증명.

---

## 5. Drift probe & 통계 (메인 분석)

### 5.1 측정
- paired drift: `d_f(c) = h_f(c) − h_text(c)` at `L*`.
- cone 좌표: `p_f(c) = [⟨d_f(c), b1⟩, ..., ⟨d_f(c), bk⟩]` (각 `bj` 단위벡터).
- family 평균 프로파일: `P_f = mean_c p_f(c)`.

### 5.2 판정 통계 (H0 vs H1)
1. **Mean pairwise cosine:** 모든 family 쌍 `(f, f')`에 대해 `cos(P_f, P_{f'})` 평균. → §0 표의 임계값과 비교.
2. **Dominant-axis 불일치:** family별 `argmax_j |P_f[j]|`가 갈리는지.
3. **Permutation test:** family 라벨을 셔플해 mean pairwise cosine의 null 분포 생성(예: 5,000회) → 관측값의 p.
4. **Bootstrap CI:** content `c`에 대해 resample해 cosine·프로파일의 95% CI.
5. (보강) **Family classifier:** `p_f(c)` k차원으로 family를 예측하는 선형 분류기 macro-F1. chance(1/4) 대비 크게 높으면 family들이 refusal-space에서 구분되는 영역을 점유한다는 직접 증거.

### 5.3 Safety Geometry Collapse 연결 (보조 metric)
- **Along-axis separation:** primary 축 `b1`에서 `proj(h_f) − proj(benign)`이 family별로 얼마나 압축되는지 → 단일 축 story와의 정량 비교(우리 결과가 그걸 **세분화**함을 보임).
- **Off-subspace drift:** k차원 cone에 projection 후 남는 잔차 norm → audio가 refusal subspace 밖으로 미는 정도(family별).

---

## 6. 산출물 (figure/table)

- **Fig 1:** layer sweep separation → `L*` 선택 근거.
- **Fig 2 (핵심):** family별 cone-축 projection 프로파일(그룹 막대 또는 radar) + family 간 pairwise cosine 히트맵. → H0/H1을 눈으로.
- **Fig 3:** family별 along-`b1` separation 압축 + off-subspace drift.
- **Table 1:** 축별 causal ablation refusal-rate 하락(축 타당성).
- **Table 2:** family별 comprehension 통과율, 유효 n, mean pairwise cosine + permutation p + bootstrap CI, classifier macro-F1.

---

## 7. 코드 스켈레톤 (구조만)

```python
# 1) 모델 로드 + config 확인
model, processor = load_qwen2_audio()  # Qwen2-Audio-7B-Instruct, bf16
L_total, D = model.config.num_hidden_layers, model.config.hidden_size

# 2) hook: 각 layer residual stream, 마지막 입력 토큰
acts = {}
def hook(layer_idx):
    def f(mod, inp, out):
        acts[layer_idx] = out[0][:, -1, :].detach().float().cpu()
    return f
register_hooks(model, hook)

def get_acts(inputs):           # text 또는 audio 입력 -> {layer: [D]}
    run_forward(model, processor, inputs)
    return dict(acts)

# 3) cone @ L* : diff-in-means(카테고리) + PCA, 그 다음 ablation 검증
H_text, B_text = load_text_pairs()        # 카테고리 라벨 포함
L_star = pick_layer_by_separation(H_text, B_text)
B_cone = build_cone(H_text, B_text, L_star)   # Gram-Schmidt + PCA cross-check
B_cone = keep_causal_axes(model, B_cone, L_star, holdout_harmful_text)  # ablation/addition

# 4) drift probe (paired)
families = ["plain", "nonspeech", "style", "perturbed"]
P = {f: [] for f in families}
for c in CONTENTS_150:
    h_text = get_acts(text_input(c))[L_star]
    if not comprehended(text=c): continue
    for f in families:
        audio = render(c, family=f)
        if not comprehended_audio(audio, c):   # 전사 일치 필터
            continue
        h_f = get_acts(audio_input(audio))[L_star]
        d = h_f - h_text
        P[f].append(project(d, B_cone))         # k차원

# 5) 통계 + 판정
profiles = {f: mean(P[f]) for f in families}
mpc = mean_pairwise_cosine(profiles)
p_perm = permutation_test(P, n=5000)
decision = decide(mpc, p_perm, dominant_axis_disagree(profiles))  # §0 표
```

---

## 8. Compute & 일정

- **Compute:** 학습 없음. 7B를 bf16로 올리고 수천 회 forward + hidden 캡처. 단일 GPU(40–80GB 권장, 24–48GB도 batch 줄이면 가능), 수 시간 규모.
- **현실적 일정(원 문서의 "7일 풀 테이블"이 아니라 1개 모델 probe):**
  - Day 1–2: 데이터 준비(카테고리 text 쌍, 4 family 렌더링, perturbation), comprehension 필터.
  - Day 2–3: activation 추출, layer sweep, cone 구성 + ablation 검증.
  - Day 3–4: drift probe + 통계 + Fig 2/3.
  - 나머지: 디버깅·재현 버퍼.

---

## 9. 첫 실험에서 일부러 **안 하는** 것 (scope 보호)

- AdvWave식 풀 adversarial 최적화 → GO 이후로.
- 다중 모델(Qwen2.5-Omni, Kimi-Audio 등) → GO 이후 generalization 단계로.
- 방어 method 구현/steering → GO 이후.
- audio-derived cone(text 대신 audio로 cone 정의) → 1차는 text-derived만(refusal이 깨끗한 곳에서 정의 + 실제 text-derived 방어 시나리오와 일치). audio-derived는 robustness 보강용으로 나중에.

---

## 10. Reviewer-proofing 체크리스트

- [ ] cone 축이 causal ablation을 통과했는가(임의 분산 축 아님)?
- [ ] family 간 content가 paired로 매칭됐는가(content confound 제거)?
- [ ] comprehension 필터로 "미인식 ≠ 안전 refusal"을 분리했는가?
- [ ] readout token 위치가 모든 조건에서 동일한가?
- [ ] style family가 다중 voice seed로 스피커 idiosyncrasy를 분산했는가?
- [ ] 판정이 사전 등록된 임계값(§0)에 따라 내려졌는가?
- [ ] permutation null + bootstrap CI로 우연이 아님을 보였는가?
- [ ] Method A(diff-in-means)와 B(PCA) subspace가 일치하는가(principal angle)?
