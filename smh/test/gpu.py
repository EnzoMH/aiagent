import torch

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"사용 가능한 GPU: {torch.cuda.device_count()}개")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA를 사용할 수 없습니다!")
        return False

if __name__ == "__main__":
    check_cuda()