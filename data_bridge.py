"""후배가 전달한 DataLoader를 연결하는 파일. Dataset 구현은 여기에 필요 없습니다.

반환값: {"train": train_loader, "val": validation_loader_or_none}
batch: mixture/speech/noise float32 [B, Z, L], 선택적으로 lengths [B].
speech_z = 목표 화자의 음성이 해당 좌석 마이크에 도달한 x_z(z).
noise_z = mixture_z - speech_z (다른 화자의 음성도 포함한 잔여 신호).
채널 순서는 항상 좌석 1, 2, 3, 4이며, 원음 s(z)와 speech_z는 다릅니다.
"""

from config import CabinSepConfig, TrainingRunConfig


def build_dataloaders(train_config: TrainingRunConfig, model_config: CabinSepConfig):
    """아래 본문을 전달받은 로더를 가져오는 코드로 바꾸세요.

    예시 (모듈명과 인자는 후배의 코드에 맞추기):

        from junior_dataset import make_loaders
        train_loader, val_loader = make_loaders(
            stage=train_config.stage, sample_rate=model_config.sample_rate
        )
        return {"train": train_loader, "val": val_loader}

    stage=1: simulated IR 데이터.
    stage=2: 발화 좌석 마이크에는 real IR, 나머지에는 simulated IR을 사용하는
             mixed IR 데이터로 미세조정 (논문 Sec. 3.4 / 4.2).
    """
    raise RuntimeError(
        "아직 실제 DataLoader가 연결되지 않았습니다. data_bridge.py의 "
        "build_dataloaders() 본문에서 후배의 로더를 반환하세요. "
        "연결 전 실행 확인은 run.py에서 MODE = 'demo'로 Run 하세요."
    )
