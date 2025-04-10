# COSE461 Team 14

## 패키지 설치

### 1. uv 설치

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```


### 2. 가상환경 생성 및 활성화

프로젝트 디렉토리로 이동하고, 가상환경을 생성:

```bash
cd ../COSE461_Project
uv venv 
```

가상환경 활성화:
```bash
source .venv/bin/activate 
```


### 3. 핵심 패키지 설치
```bash
uv sync        # 기본 패키지 설치
uv sync --dev  # 개발용 패키지 설치
```


## 프로젝트 모니터링

### 1. wandb 로그인

초대를 받은 후 아래 명령어로 로그인:

```bash
wandb login
```

### 2. 프로젝트 초기화

wandb를 사용하여 프로젝트를 초기화:

```python
import wandb
run = wandb.init(project="team14")
```

### 3. 데이터 기록

wandb를 사용하여 학습 중의 메트릭(예: 손실, 정확도 등)을 기록:

```python
# 학습 루프 내에서 로그 기록
run.log({
    "epoch": epoch,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "train_acc": train_acc,
    "val_acc": val_acc
})
```

### 4. 대시보드 확인

로그가 기록되면 wandb 웹 대시보드에서 실시간으로 데이터를 확인할 수 있음.


## 프로젝트 레포트

https://www.overleaf.com/8765536775bnhbqzgmgnkc#54c42d