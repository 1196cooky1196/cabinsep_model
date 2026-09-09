# CabinSep 논문 대조 및 구현 기록

기준은 첨부된 [CabinSep 논문](10_CabinSep.pdf), 특히 Figure 1과 식 (1)–(7)입니다.
논문에 명시된 내용은 PAPER, 인용 논문에서 가져온 설정은 REFERENCE,
공개된 설명으로 확정할 수 없는 구현 선택은 ASSUMPTION으로 구분합니다.
파라미터 수가 비슷하다는 이유만으로 층의 축이나 차원을 확정하지 않습니다.

## 현재 완료 범위

- S/M/L 모델, Spec/LPS/IPD, Full-band LSTM, TAC, Sub-band Conformer, speech/noise dual mask.
- MVDR를 제외한 mask 기반 학습, 식 (7) loss, 검증, Adam/step LR, stage 2 및 학습 재개.
- 외부 DataLoader 연결 함수, 가변 길이와 layout 어댑터.
- checkpoint 기반 전체 파일 추론, MVDR와 WAV/mask/manifest 저장.
- Run 버튼용 demo/check/train/inference 진입점.
- CPU 회귀 검사 30개, 실제 S 모델 2-step demo와 WAV 4개 저장, S/M/L 짧은 입력 추론 통과.

실제 데이터로 본학습하거나 CER/NSPA를 측정하지 않았습니다. 논문의 GMAC/차량 RTF도
측정하지 않았습니다. GPU는 인식되지만 추가 실행 확인 중 CUDA 초기화/메모리 할당 오류가
발생하여 GPU 학습 검증을 완료하지 못했습니다. 실행 중인 다른 프로그램이나 시스템 설정은
변경하지 않았습니다.

## 논문과 코드 대응

| 구성 | 코드 | 근거 및 선택 |
|---|---|---|
| STFT | `CabinSep.stft/istft` | PAPER: Hamming, FFT 512, window 32 ms, hop 16 ms. ASSUMPTION: 16 kHz, center=False, periodic window, 끝부분 zero pad와 수동 overlap-add. |
| Spec | `extract_features/SpecEncoder` | PAPER: 복소 spectrum 실수/허수 적층 `[B,2Z,T,F]`, Conv-ReLU 2회. |
| LPS | `extract_features/LPSEncoder` | `log(real²+imag²+epsilon)`, `[B,Z,T,F]`. 본문의 Y_R 채널 표기 모순은 아래 참조. |
| IPD | `extract_features/IPDEncoder` | PAPER: 앞좌석 1/2의 phase 차이 cos/sin, `[B,2,T,F]`. |
| Encoder 폭/kernel | `config.py` | ASSUMPTION: 각각 폭 8, kernel 1×3. 논문은 이 수치를 주지 않음. |
| Fusion | `CabinSep.fusion` | PAPER: 출력 C=24. kernel 1×3은 REFERENCE/ASSUMPTION. |
| Full-band | `FullBandLSTM` | REFERENCE [30]: 주파수 downsample → flatten → PReLU/cGLN → 단방향 LSTM → Linear/cGLN/PReLU → deconv → residual. |
| TAC | `TACBlock` | 본문 식 (6)은 C→C/d이나 Table 1의 약 0.40 M 감소는 F→F/d와 일치. 재현 기본값은 frequency이며 literal channel도 설정 가능. |
| Time skip | `FullSubModule` | 선택 옵션. 기본 False는 Table 1 base에 대응. True는 random training parity / deterministic eval parity로 절반 프레임 처리. |
| Sub-band | `SubBandConformer` | PAPER/REFERENCE: downsample → `[B*Fsub,T,H]` shared Conformer → deconv/residual. H=16, FFN=8, heads=4는 PAPER. |
| Conformer attention | `CausalConformerBlock` | REFERENCE [32]: Transformer-XL 상대 sinusoidal 위치 정보, causal mask. legacy는 위치 정보 없는 MHA. |
| Mask head | `MaskEstimator` | PAPER: speech LN→Linear→Sigmoid, noise LN→Linear→GLU→Linear→ReLU. F축과 noise hidden=F는 ASSUMPTION. |
| 학습 경로 | `forward_train` | PAPER: zone별 mask × 해당 microphone spectrum → iSTFT. MVDR는 학습하지 않음. |
| Loss | `CabinSepLoss` | PAPER 식 (7): speech FBank MAE×0.01 + negative SI-SNR + noise FBank MAE×0.01. FBank 세부 설정은 ASSUMPTION. |
| 추론 | `StreamingMVDR/separate` | 식 (3)/(4)에 따른 reference-channel MVDR. 누적 covariance, 초기화/안정화는 ASSUMPTION. |
| Optimizer | `run_training` | PAPER: Adam 1e-4, optimizer update 20,000번마다 LR 절반. |
| 두 단계 학습 | `stage/stage1_checkpoint` | PAPER: simulated IR 학습 후 mixed real/simulated IR 미세조정. Stage 2 Adam 초기화는 ASSUMPTION. |

정규화된 float32 파형 입출력은 `[B,Z,L]`, 복소 spectrum과 mask는 `[B,Z,T,F]`입니다.
기본 Z=4, F=257이며 앞좌석 microphone 인덱스는 Python에서 (0,1)입니다.

## 논문의 모호한 부분과 선택 이유

### TAC 축

식 (6)은 Linear A/B의 압축을 C→C/d로 명시합니다. 그러나 C=24인 channel Linear는
CabinSep-L의 세 TAC를 모두 합쳐도 수천 parameter뿐입니다. Table 1에서 conformer 제거 후
TAC까지 제거할 때 줄어드는 약 0.40 M은 F=257에 F→F/d Linear A/B/C를 적용하면 약 0.396 M로
맞습니다. 실제 보고 모델을 재현하는 기본값은 `tac_axis="frequency"`로 정했습니다.

이때 Linear A/B는 F를 압축하고 branch B는 C축으로 평균·반복한 뒤 A/B를 F축으로 concat합니다.
식 자체를 그대로 실험하려면 `tac_axis="channel"`을 사용합니다. 논문 내부 모순이므로 어느
한쪽을 공개된 사실이라고 단정하지 않으며 두 모델의 가중치는 서로 호환되지 않습니다.

### Mask head

Figure 1(e)의 분기 순서는 명확하지만 Linear 및 LN 축은 제시하지 않습니다.
F→2F→GLU→F→F는 noise-head ablation의 파라미터 감소와 대략 맞는 가정입니다.
이를 논문에 명시된 차원이라고 표현하지 않습니다.

### Time skip

Figure 1과 본문에서는 구조의 일부처럼 설명하지만 Table 1에서는 `+time skip` ablation을
별도 보고합니다. 기본 `time_skip=False`와 선택 가능한 True를 모두 제공합니다.
True일 때 training parity는 batch 단위 무작위, eval은 `inference_time_skip_offset`입니다.
odd parity에서 1-frame 입력을 임의로 even parity로 바꾸던 경계 오류를 수정했습니다.

### LPS 및 MVDR 기호

LPS 본문은 2Z채널 Y_R을 쓰면서 결과를 Z채널로 표현합니다. 복소 spectrum의
`real²+imag²`를 사용하는 해석이 차원과 일치합니다.

식 (4) 뒤 target/interference 설명 순서도 일반적인 MVDR 수식과 어긋납니다.
이 코드는 아래 순서를 사용합니다.

```text
Psi = noise/interference mask로 가중한 covariance
Phi = speech/target mask로 가중한 covariance
A   = solve(Psi + diagonal_loading*I, Phi)
w   = A @ reference / trace(A)
X   = wᴴ @ Y
```

좌석 z의 reference microphone은 z입니다. 목표 x_z(z)와 일치하는 선택이지만 reference
선택 자체는 논문에서 구체적으로 설명하지 않습니다.

## 참조 논문에서 가져온 부분

[FSB-LSTM, 참조 [30]](https://arxiv.org/html/2304.08707)의 full-band 설정은
Conv 출력 E=8, kernel=8, stride=4, LSTM hidden=256이며 sub-band kernel/stride는 5/5입니다.
CabinSep가 참조 구조를 채택한다고 설명하므로 가져왔지만 수치까지 동일하다는 보장은 없습니다.
참조 구현의 custom frequency overlap-add 대신 ConvTranspose2d를 사용하므로 MAC 수를
직접 비교해서는 안 됩니다.

[Conformer, 참조 [32]](https://arxiv.org/html/2005.08100v1#S2.SS1)의 상대 위치 attention을
추가했습니다. score는 `((Q+u)Kᵀ + (Q+v)R_(query-key)ᵀ)/sqrt(head_dim)`입니다.
별도의 scalar 계산식과 비교하는 회귀 검사로 구현을 확인했습니다.

Macaron FFN, GLU, causal depthwise convolution, SiLU를 사용합니다. CabinSep가 상세 변경을
설명하지 않으므로 인용된 원 Conformer [32]를 따라 convolution BatchNorm, kernel=32,
dropout=0.1을 기본값으로 사용합니다. 추론 시 BatchNorm은 저장된 running statistics를 써서
미래 프레임을 읽지 않습니다. 이전 per-frame LayerNorm 모델은 `conformer_conv_norm="layer"`로
복원할 수 있습니다. convolution은 논문의 causal 표기에 맞게 왼쪽 padding만 적용합니다.
`conformer_left_context_frames=125`는 16 ms hop에서 현재 프레임을 포함해 약 2초 범위입니다.
이 옵션을 켜더라도 현재 구현은 전체 attention 행렬을 만들며 KV cache를 제공하지 않습니다.

## MVDR 수치 안정성과 streaming 범위

[참조 [31]](https://ieeexplore.ieee.org/document/8461850)은 frame-wise inverse 갱신 방법을
제안합니다. CabinSep는 dual-mask update, 초기화, forgetting factor를 모두 기술하지 않습니다.
현재 구현은 가중 covariance 누적과 Cholesky 기반 선형계 풀이를 사용하며
[31]의 Woodbury inverse update를 그대로 재현하지 않습니다.

동일한 microphone 신호 등 rank가 낮은 입력에서 diagonal loading이 float 정밀도 때문에
사라지지 않도록 상대 크기 하한을 둡니다. 행렬별 factorization 실패나 유효 speech mask가
없는 경우 해당 reference microphone으로 fallback합니다. 무음 좌석을 별도 VAD로
억제하는 기능은 없으며 fallback 구간에는 원 microphone 신호가 남을 수 있습니다.

`StreamingMVDR`의 covariance 상태는 spectrum chunk 사이에 전달할 수 있으며 전체
spectrum 처리 결과와 일치하는 검사를 통과했습니다. `CabinSep.separate`의 네트워크는
전체 입력 파일을 한 번에 처리합니다. MVDR state만 전달해도 LSTM/attention/STFT 상태가
이어지는 것은 아닙니다. 연속 raw waveform streaming API, cache와 지연 측정은 미구현입니다.

## Loss와 DataLoader 계약

FBank: 80 mel bands, 25 ms Hamming window, 10 ms hop, FFT 512, 20 Hz–Nyquist,
log(power+epsilon)입니다. 논문이 설정을 공개하지 않아 선택한 값이며 torchaudio는 필요 없습니다.
win_length<n_fft인 center=False STFT가 처음과 마지막 샘플 일부를 감독하지 않던 문제를
피하려고 실제 window 길이로 frame을 만든 뒤 FFT zero-padding을 합니다. 마지막 부분 frame도
포함합니다.

SI-SNR은 zero-mean입니다. target energy가 없는 좌석은 SI-SNR 계산에서 제외하고
FBank로 감독합니다. 활성 좌석 평균을 예제마다 계산한 뒤 batch 평균을 취합니다.
모든 좌석이 무음인 예제의 SI-SNR 항은 0입니다. 이 reduction도 논문에 없는 선택입니다.
좌석 정렬 정답을 사용하므로 PIT는 적용하지 않습니다.

`lengths[B]`가 있으면 adapter는 패딩을 0으로 만들고, 같은 길이의 예제를 묶어 실제 길이로
crop한 뒤 모델 STFT부터 실행합니다. loss도 같은 실제 길이만 사용합니다. 따라서 짧은 음성을
단독 처리한 결과와 padded batch의 유효 구간이 일치하고 패딩의 loss gradient는 0입니다.

데이터 담당자가 반드시 맞춰야 하는 의미:

- mixture_z: 해당 microphone의 혼합 신호.
- speech_z: 목표 화자의 해당 microphone 도달 신호 x_z(z), dry s(z)가 아님.
- noise_z: mixture_z−speech_z, 다른 화자의 음성을 포함한 간섭.
- 좌석 순서와 sample rate는 학습과 추론에서 동일.
- Stage 2 mixed IR: 발화 좌석 microphone에는 real IR, 다른 microphone에는 simulated IR.

Dataset 생성, IR 측정/증강, sampling, SNR 범위, train/val/test 분리는 데이터 담당자의 범위입니다.
모델의 stage 숫자를 변경하는 것만으로 IR 증강이 수행되지는 않습니다.

## Checkpoint와 재개

새 Run 학습은 `config.py`의 `TRAIN_MODEL`, `TRAIN_LOSS`, `TRAIN_RUN`을 사용합니다.
따라서 S/M/L뿐 아니라 TAC 축, time skip, Conformer context 등도 터미널 인자 없이
명시적으로 선택할 수 있습니다.

새 checkpoint는 모델/loss/run 설정, 모델 가중치, Adam, scheduler, 총 step, 다음 epoch/batch,
Python/NumPy/Torch/CUDA RNG 및 접근 가능한 DataLoader/sampler generator 상태를 저장합니다.
임시 파일에 쓴 후 교체합니다. 검증 후 periodic checkpoint를 저장하여 best metric이 뒤처지지
않게 했습니다. NaN/Inf loss 또는 gradient는 진단 checkpoint를 쓰고 중단합니다.

재개는 이미 처리한 batch를 다시 optimizer에 넣지 않으며 max_steps/epochs 총 한도를
초과하지 않습니다. deterministic shuffled loader를 사용한 CPU 검사에서 연속 6-step 학습과
2-step 후 중단/재개한 모델 가중치가 정확히 일치했습니다. 임의 external state, persistent
worker RNG, 변경된 데이터셋/loader까지 동일하게 재생한다는 의미는 아닙니다.

구 checkpoint에서 새 옵션이 없으면 frequency TAC, time skip=True, 상대 위치=False로 읽고,
state key에 따라 BatchNorm/LayerNorm을 선택합니다. 기존 `tmp/integration_check`의 BatchNorm
checkpoint도 strict load 및 유한 추론을 확인했습니다.
기존 checkpoint로 재개해도 FBank 경계와 SI-SNR reduction은 현재 수정된 loss를 사용합니다.

## 파라미터 수

수정된 기본값(frequency TAC, time skip=False, 상대 위치=True, Conformer [32] 기본)의
실제 trainable count입니다.

| 모델 | 현재 코드 | 논문 Table 1 |
|---|---:|---:|
| S | 1,279,489 | 약 1.09 M |
| M | 2,275,834 | 약 2.24 M |
| L | 3,476,883 | 약 3.43 M |

논문과 수치가 일치하지 않습니다. 공개되지 않은 encoder/Conformer/head 차원, 참조 구조의
세부 설정과 TAC 설명의 모호함이 남아 있습니다. 수치를 맞추기 위한 임의 수정 대신 논문에
명시된 수식과 가정을 추적할 수 있는 베이스라인으로 제공합니다.
