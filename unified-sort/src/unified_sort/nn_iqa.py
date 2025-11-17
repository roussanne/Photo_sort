"""
Deep Learning based No-Reference Image Quality Assessment (NR-IQA).

딥러닝 기반 무참조 이미지 품질 평가 모듈:
- PyTorch 기반 CNN 품질 평가
- NIMA(Neural Image Assessment) 스타일 아키텍처
- 블러 타입 분류 (Sharp/Defocus/Motion)
- 전통적 방법과 융합 가능

개선사항:
1. 선택적 의존성 (PyTorch 없어도 작동)
2. 경량 모델 (빠른 추론)
3. 스레드 안전한 싱글톤
4. 타입 힌트 및 검증
"""

from typing import Optional, Dict, Tuple, Any
from pathlib import Path
import numpy as np
import threading


# 전역 모델 인스턴스 (싱글톤)
_model_instance: Optional['NNQuality'] = None
_model_lock = threading.Lock()


# PyTorch 가용성 확인
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# Base class selection based on PyTorch availability
if _TORCH_AVAILABLE:
    _BaseModel = nn.Module
else:
    _BaseModel = object


class SimpleCNN(_BaseModel):
    """
    경량 CNN 모델 for 블러 분류.

    NIMA-inspired architecture with reduced complexity:
    - 3 convolutional layers
    - Global average pooling
    - 3-class output (sharp, defocus, motion)
    """

    def __init__(self, num_classes: int = 3):
        """
        Args:
            num_classes: 출력 클래스 수 (기본값 3: sharp/defocus/motion)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SimpleCNN")

        super(SimpleCNN, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Output logits (B, num_classes)
        """
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class NNQuality:
    """
    딥러닝 기반 이미지 품질 평가기.

    스레드 안전한 싱글톤 패턴으로 구현되어 메모리 효율적입니다.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            model_path: 사전 학습된 모델 가중치 경로 (None이면 랜덤 초기화)
            device: 'cpu', 'cuda', 또는 None (자동 선택)
            input_size: 입력 이미지 크기 (H, W)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for deep learning quality assessment. "
                "Install with: pip install torch torchvision"
            )

        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.input_size = input_size

        # 모델 초기화
        self.model = SimpleCNN(num_classes=3)

        # 가중치 로드
        if model_path and Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model weights: {e}")
                print("Using randomly initialized weights")
        else:
            if model_path:
                print(f"Warning: Model path {model_path} not found")
            print("Using randomly initialized weights (for demonstration)")

        self.model.to(self.device)
        self.model.eval()

        # 정규화 파라미터 (ImageNet 표준)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, image: np.ndarray) -> 'torch.Tensor':
        """
        이미지 전처리.

        Args:
            image: BGR 이미지 (numpy array, uint8)

        Returns:
            전처리된 텐서 (1, 3, H, W)
        """
        import cv2

        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image_resized = cv2.resize(image_rgb, self.input_size, interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        image_float = image_resized.astype(np.float32) / 255.0

        # To tensor (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)

        # ImageNet normalization
        image_tensor = image_tensor.to(self.device)
        image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor

    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        이미지 품질 예측.

        Args:
            image: BGR 이미지 (numpy array)

        Returns:
            품질 점수 딕셔너리 {sharp_score, defocus_score, motion_score}
        """
        try:
            with torch.no_grad():
                # 전처리
                input_tensor = self.preprocess(image)

                # 추론
                logits = self.model(input_tensor)

                # Softmax for probabilities
                probs = F.softmax(logits, dim=1)

                # CPU로 이동 및 numpy 변환
                probs_np = probs.cpu().numpy()[0]

                # 결과 반환
                return {
                    "sharp_score": float(probs_np[0]),
                    "defocus_score": float(probs_np[1]),
                    "motion_score": float(probs_np[2])
                }

        except Exception as e:
            print(f"Warning: Deep learning prediction failed: {e}")
            # 폴백: 균등 분포
            return {
                "sharp_score": 0.33,
                "defocus_score": 0.33,
                "motion_score": 0.34
            }

    def unload(self):
        """모델을 메모리에서 언로드합니다."""
        if hasattr(self, 'model'):
            del self.model
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_model(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    force_reload: bool = False
) -> Optional[NNQuality]:
    """
    싱글톤 모델 인스턴스를 가져옵니다.

    스레드 안전하게 구현되어 있습니다.

    Args:
        model_path: 모델 가중치 경로
        device: 'cpu' 또는 'cuda'
        force_reload: 강제로 새 인스턴스 생성

    Returns:
        NNQuality 인스턴스, PyTorch 없으면 None
    """
    global _model_instance

    if not _TORCH_AVAILABLE:
        return None

    with _model_lock:
        if _model_instance is None or force_reload:
            try:
                _model_instance = NNQuality(
                    model_path=model_path,
                    device=device
                )
            except Exception as e:
                print(f"Warning: Failed to initialize NNQuality: {e}")
                _model_instance = None

        return _model_instance


def unload_model():
    """
    전역 모델 인스턴스를 언로드합니다.

    메모리 절약을 위해 사용합니다.
    """
    global _model_instance

    with _model_lock:
        if _model_instance is not None:
            _model_instance.unload()
            _model_instance = None


def predict_quality(
    image: np.ndarray,
    model_path: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    이미지 품질을 예측합니다 (편의 함수).

    Args:
        image: BGR 이미지 (numpy array)
        model_path: 모델 가중치 경로 (선택)
        device: 'cpu' 또는 'cuda' (선택)

    Returns:
        품질 점수 딕셔너리
    """
    model = get_model(model_path=model_path, device=device)

    if model is None:
        # PyTorch 없음 - 폴백
        return {
            "sharp_score": 0.33,
            "defocus_score": 0.33,
            "motion_score": 0.34
        }

    return model.predict(image)


def fuse_scores(
    signal_scores: Dict[str, float],
    dl_scores: Dict[str, float],
    dl_weight: float = 0.3
) -> Dict[str, float]:
    """
    신호 처리 기반 점수와 딥러닝 점수를 융합합니다.

    Args:
        signal_scores: 전통적 방법의 점수 (VoL, Tenengrad, etc.)
        dl_scores: 딥러닝 모델의 점수
        dl_weight: 딥러닝 점수의 가중치 (0.0 ~ 1.0)

    Returns:
        융합된 점수 딕셔너리
    """
    # 가중치 검증
    dl_weight = max(0.0, min(1.0, dl_weight))
    signal_weight = 1.0 - dl_weight

    # 융합
    fused = {}
    for key in ["sharp_score", "defocus_score", "motion_score"]:
        signal_val = signal_scores.get(key, 0.33)
        dl_val = dl_scores.get(key, 0.33)

        fused[key] = signal_weight * signal_val + dl_weight * dl_val

    # 정규화 (합이 1이 되도록)
    total = sum(fused.values())
    if total > 0:
        for key in fused:
            fused[key] /= total

    return fused


def is_available() -> bool:
    """
    딥러닝 기능 사용 가능 여부를 확인합니다.

    Returns:
        PyTorch가 설치되어 있으면 True
    """
    return _TORCH_AVAILABLE


def get_device_info() -> Dict[str, Any]:
    """
    현재 사용 가능한 디바이스 정보를 반환합니다.

    Returns:
        디바이스 정보 딕셔너리
    """
    info = {
        "torch_available": _TORCH_AVAILABLE,
        "cuda_available": False,
        "device_count": 0,
        "current_device": "cpu"
    }

    if _TORCH_AVAILABLE and torch is not None:
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = "cuda"

    return info
