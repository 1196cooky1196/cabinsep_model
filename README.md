# CabinSep 학습·추론 코드

논문 `10_CabinSep.pdf`를 바탕으로 구현한 PyTorch 연구용 베이스라인입니다.
모델, loss, 학습, 검증, checkpoint 저장/재개, MVDR 추론과 좌석별 WAV 저장을 제공합니다.
Dataset 생성과 IR 증강은 데이터 담당자의 로더에서 수행합니다.

## Run 버튼으로 시작

**`run.py`를 열고 우측 상단 Run을 누르세요.** 기본 `MODE = "demo"`는 데이터나
GPU 없이 합성 파형으로 2-step 학습 → checkpoint 재로드 → 추론 → 4개 WAV 저장을
실행합니다. 출력은 `outputs/demo/<실행 시각>/`입니다.
합성 잡음을 이용한 실행 검사이므로 이 WAV로 음성 분리 성능을 평가하면 안 됩니다.

| run.py의 MODE | 실행 내용 | 먼저 준비할 것 |
|---|---|---|
| `"demo"` | 합성 데이터로 짧은 학습과 파일 추론 | 없음 |
| `"train"` | 실제 데이터로 학습 | `data_bridge.py`, `config.py`의 `TRAIN_RUN` |
| `"inference"` | 저장한 모델로 좌석별 WAV 생성 | `config.py`의 `INFERENCE_RUN` |
| `"check"` | 모델·loss·역전파·인과성 검사 | 없음 |

`model.py` 또는 `smoke_test.py`를 직접 Run해도 모델 검사가 실행됩니다.
`train.py`, `inference.py`도 각각 직접 Run할 수 있습니다. 터미널 인자는 필요 없습니다.
상대 경로는 작업 디렉터리와 관계없이 이 프로젝트 폴더를 기준으로 해석합니다.

## 후배의 DataLoader 연결

`data_bridge.py`의 `build_dataloaders()`에서 전달받은 로더를 반환하면 됩니다.
실제 연결 예제가 함수 안에 있습니다. 기본 반환 형식은 다음과 같습니다.

```python
return {"train": train_loader, "val": val_loader}  # 검증 로더가 없으면 None
```

로더는 아래 batch를 반환해야 합니다. 원시 PCM 정수 대신 정규화된 실수 파형을 사용합니다.

```python
{
    "mixture": FloatTensor[B, 4, L],
    "speech":  FloatTensor[B, 4, L],
    "noise":   FloatTensor[B, 4, L],
    "lengths": LongTensor[B],  # 가변 길이를 패딩한 경우 실제 샘플 수
    "sample_rate": 16000,     # 선택 사항; 있으면 모델 설정과 일치하는지 검사
}
```

- `B`: batch 크기, `L`: 샘플 수. 채널 순서는 좌석 1, 2, 3, 4이며 앞좌석은 1, 2입니다.
- **speech_z는 해당 좌석 마이크에 도달한 목표 음성 `x_z(z)`입니다.** 무잔향 원음
  `s(z)`를 그대로 넣으면 논문의 학습 타깃과 달라집니다.
- **noise_z는 `mixture_z - speech_z`**에 해당합니다. 다른 화자의 음성과 배경 소음을
  모두 포함해야 합니다. 배경 소음만을 의미하지 않습니다.
- 모든 예제가 동일한 유효 길이라면 `lengths`를 생략합니다. 패딩 배치에는 반드시
  실제 길이를 전달하세요. 패딩 부분은 입력에서 0으로 만들고 loss 계산에서 제외합니다.
- 키 이름이 다르면 `TRAIN_RUN`의 `mixture_key`, `speech_key`, `noise_key`,
  `lengths_key`를 변경합니다. `[B,L,4]` 형태는 `batch_layout="BLZ"`로 연결합니다.
- 반복 가능한 DataLoader를 넘기세요. `iter(loader)` 같은 일회성 iterator는 지원하지 않습니다.

이미 만들어진 로더를 Python에서 직접 전달하는 것도 가능합니다.

```python
from config import TRAIN_RUN, CabinSepConfig, LossConfig
from train import run_training

latest_checkpoint = run_training(
    TRAIN_RUN, train_loader=train_loader, validation_loader=val_loader,
    model_config=CabinSepConfig.small(), loss_config=LossConfig(),
)
```

## 학습 설정과 저장

`config.py` 맨 아래 `TRAIN_MODEL`, `TRAIN_LOSS`, `TRAIN_RUN`에 원하는 값을 지정합니다.

```python
TRAIN_MODEL = CabinSepConfig.small(
    time_skip=False,          # Table 1 기본 모델; +time skip 실험은 True
    tac_axis="frequency",     # Table 1 파라미터 수 기반 재현값
)
TRAIN_LOSS = LossConfig()
TRAIN_RUN = TrainingRunConfig(
    model_variant=TRAIN_MODEL.variant,
    stage=1,
    run_name="cabinsep_s_stage1",
    device="auto",            # CUDA가 있으면 CUDA, 없으면 CPU
    epochs=100,
)
```

M/L 모델은 각각 `CabinSepConfig.medium()`, `CabinSepConfig.large()`로 바꿉니다.
checkpoint 재개와 Stage 2에서는 저장된 모델·loss 설정이 우선하므로 새 설정을 섞지 않습니다.

논문의 Adam 초기 학습률 `1e-4`, **optimizer 20,000 step마다 LR 절반**을 반영합니다.
batch 크기, 음성 길이, augmentation과 샘플링은 후배의 DataLoader에서 설정합니다.
새 실험에는 새 `run_name`을 사용합니다. 기존 실험 폴더에 새 학습을 덮어쓰지 않습니다.

`checkpoints/<run_name>/`에 `latest.pt`, `best.pt`, 주기적 `step_*.pt`,
`losses.jsonl`이 저장됩니다. `best.pt`는 검증 loss 최솟값 모델입니다.
검증 로더가 없으면 `best.pt`도 마지막 모델이며, 일반화 성능을 선택한 모델은 아닙니다.

Stage 2는 아래처럼 설정합니다. 로더는 `train_config.stage`를 보고 stage에 맞는 데이터를
선택해야 합니다. 논문의 mixed IR 전략은 **발화 좌석 마이크에 real IR, 나머지 마이크에
simulated IR**을 사용합니다. 모델 코드에서 데이터에 IR을 다시 적용하지 않습니다.

```python
TRAIN_RUN = TrainingRunConfig(
    stage=2,
    stage1_checkpoint="checkpoints/cabinsep_s_stage1/best.pt",
    run_name="cabinsep_s_stage2",
)
```

중단한 학습은 `resume_checkpoint="checkpoints/<run_name>/latest.pt"`와 해당
`run_name`, `stage`를 지정합니다. 모델·loss 설정, Adam·scheduler, step과 다음 batch
위치를 복원합니다. `epochs`와 `max_steps`는 **재개 전 학습을 포함한 총 한도**입니다.
stage 2 시작은 가중치를 이어받고 optimizer와 step은 새로 시작합니다.
외부 데이터셋의 임의 상태나 persistent worker 내부 RNG까지 완전히 복원하지는 못합니다.

## 추론 결과

`INFERENCE_RUN`에서 checkpoint와 입력 다채널 WAV를 지정하고 `MODE="inference"`로
Run하세요. 기본 경로는 다음과 같습니다.

```python
INFERENCE_RUN = InferenceRunConfig(
    checkpoint_path="checkpoints/cabinsep_s_stage1/best.pt",
    input_path="input/multichannel.wav",
    output_dir="outputs/inference",
)
```

입력은 checkpoint와 동일한 sample rate 및 좌석 채널 순서를 사용해야 합니다.
결과는 `zone_1.wav` ~ `zone_4.wav`, `masks.pt`, `manifest.json`입니다.
WAV는 clipping을 피하도록 float32로 저장합니다. 각 좌석의 분리된 음성 파형을 출력하며
텍스트가 필요하면 이 WAV를 별도 ASR에 전달해야 합니다.

코드에서는 `model.eval()`과 `torch.inference_mode()` 안에서
`model.separate(mixture)["waveform"]`을 호출할 수 있습니다. 입출력은 `[B,4,L]`입니다.
현재 추론은 **파일 전체를 처리하는 causal 모델**입니다. 녹음 길이가 길어지면 attention
메모리가 크게 증가하므로 짧은 발화 파일로 먼저 실행하세요. 연속 마이크 입력에 대한
STFT/LSTM/attention 상태 캐시와 차량 실시간 배포는 별도 구현이 필요합니다.

## 논문 대조 및 검증

`IMPLEMENTATION_NOTES.md`에 논문 수식별 대응과 명시되지 않은 구현 선택을 기록했습니다.
이번 검토에서는 TAC의 수식/Table 1 모순을 설정으로 분리하고 재현 기본값은 보고된
파라미터 수에 맞는 주파수 축으로 정했습니다. time skip을 선택 옵션으로 만들고,
인용된 Conformer의 상대 위치 정보·BatchNorm·kernel 32·dropout 0.1을 반영했습니다.
패딩 loss, 짧은 파형의 FBank 경계,
중간 checkpoint 재개와 best 모델 저장 처리도 보완했습니다.

이전 checkpoint는 저장된 설정과 state key를 보고 TAC, time skip, attention 및
Conformer normalization 구조를 복원합니다. 기존 `tmp/integration_check`의 BatchNorm
checkpoint도 strict load와 유한 추론을 확인했습니다.
수정된 구조로 새 실험을 하려면 checkpoint 없이 새 `run_name`으로 학습하세요.
기존 checkpoint 재개 시 loss의 경계 처리와 평균 방식은 수정된 구현을 사용합니다.

`test_model.py`, `test_loss.py`, `test_training.py`는 각각 Run 버튼으로 실행하는
회귀 검사입니다. 실제 차량 데이터 학습 성능, CER/NSPA, 논문의 GMAC/RTF는 아직 검증하지
않았습니다. 논문에 없는 차원 등을 가정했으므로 공식 구현과 완전히 동일하다는 의미는 아닙니다.

## Python 환경

Python 3.10 이상, PyTorch, NumPy, SoundFile을 사용합니다. 신규 환경의 권장 의존성은
`requirements.txt`에 있습니다. 편집기에서 이 패키지들이 설치된 Python 인터프리터를
선택하세요. 현재 PC에서 확인한 환경은 Python 3.10, PyTorch 1.12.1+cu113,
GTX 1660 SUPER(6 GB)입니다. 기존 환경은 변경하지 않았습니다.
CPU 회귀 검사 30개와 demo의 학습·파일 추론은 통과했습니다. GPU 추가 확인은 이 환경의
CUDA 초기화/메모리 할당 오류로 완료하지 못했습니다. GPU 실행이 실패하면 `device="cpu"`로
코드 동작을 먼저 확인하고, 본학습 전에 CUDA 환경과 메모리 상태를 점검해야 합니다.
