"""이 파일을 열고 우측 상단 Run을 누르세요. 터미널 인자는 필요 없습니다.

MODE만 선택하고 실제 학습/추론 경로와 설정은 config.py에서 변경합니다.
demo: 데이터 없이 2-step 시험 학습 -> checkpoint -> 좌석별 WAV 저장.
train: data_bridge.py에 연결한 실제 DataLoader로 학습.
inference: INFERENCE_RUN의 checkpoint와 다채널 WAV로 추론.
check: 모델 구조, loss, backward, causality 검증.
"""

from pathlib import Path


MODE = "demo"  # "demo", "train", "inference", "check" 중 하나


def run_demo() -> Path:
    from datetime import datetime
    import torch
    import soundfile as sf

    from config import CabinSepConfig, InferenceRunConfig, TrainingRunConfig
    from inference import run_inference
    from train import run_training

    # 실제 데이터 로더를 대체하지 않는, 실행 확인용 합성 파형입니다.
    # 재실행할 때 기존 결과가 덮어써지지 않도록 매번 새 폴더를 만듭니다.
    output_root = Path(__file__).resolve().parent / "outputs" / "demo"
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    generator = torch.Generator().manual_seed(7)
    model_config = CabinSepConfig.small()
    samples = 2048

    def toy_batch():
        speech = 0.1 * torch.randn(1, 4, samples, generator=generator)
        # 서로 다른 좌석 화자의 누설과 배경 잡음을 섞은 간단한 계산 예제.
        noise = 0.03 * torch.randn(1, 4, samples, generator=generator)
        noise = noise + 0.2 * speech.roll(1, dims=1)
        return {"mixture": speech + noise, "speech": speech, "noise": noise,
                "sample_rate": model_config.sample_rate}

    training_batches = [toy_batch(), toy_batch()]
    validation_batches = [toy_batch()]
    print("DEMO: synthetic audio, 2 optimizer steps. This does not measure separation quality.")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        latest_checkpoint = run_training(
            TrainingRunConfig(
                checkpoint_dir=str(output_root), run_name=run_name, device="cpu",
                epochs=1, max_steps=2, log_every_steps=1,
                validate_every_steps=2, checkpoint_every_steps=2,
            ),
            train_loader=training_batches, validation_loader=validation_batches,
            model_config=model_config,
        )
        checkpoint_directory = latest_checkpoint.parent
        input_path = checkpoint_directory / "demo_input.wav"
        sf.write(input_path, validation_batches[0]["mixture"][0].T.numpy(),
                 model_config.sample_rate, subtype="FLOAT")
        result = run_inference(InferenceRunConfig(
            checkpoint_path=str(checkpoint_directory / "best.pt"),
            input_path=str(input_path), output_dir=str(checkpoint_directory / "separated"),
            device="cpu",
        ))
    finally:
        torch.set_num_threads(previous_threads)
    print(f"DEMO complete. Synthetic test outputs: {result}")
    return result


def main() -> None:
    if MODE == "demo":
        run_demo()
    elif MODE == "train":
        from train import main as train_main
        train_main()
    elif MODE == "inference":
        from inference import main as inference_main
        inference_main()
    elif MODE == "check":
        from smoke_test import main as check_main
        check_main()
    else:
        raise ValueError("MODE must be demo, train, inference, or check")


if __name__ == "__main__":
    # Required on Windows when the supplied DataLoader uses worker processes.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
