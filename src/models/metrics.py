import torch
from torch.utils.data import DataLoader

Tensor = torch.Tensor


class Metric:
    def __repr__(self) -> str:
        raise NotImplementedError

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        raise NotImplementedError


class MASE(Metric):
    def __init__(self, dataloader: DataLoader, horizon: int) -> None:
        self.scale = self.naivenorm(dataloader, horizon)

    def __repr__(self) -> str:
        return f"MASE(scale={self.scale:.3f})"

    def naivenorm(self, dataloader: DataLoader, horizon: int) -> Tensor:
        elist = []
        for x, y in dataloader:
            yhat = self.naivepredict(x, horizon)
            e = self.mae(y, yhat)
            elist.append(e)
        return torch.mean(torch.tensor(elist))

    def naivepredict(self, x: Tensor, horizon: int) -> Tensor:
        assert horizon > 0
        yhat = x[..., -horizon:, :].squeeze(-1)
        return yhat

    def mae(self, y: Tensor, yhat: Tensor) -> Tensor:
        return torch.mean(torch.abs(y - yhat))

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        return self.mae(y, yhat) / self.scale


class MAE(Metric):
    def __repr__(self) -> str:
        return "MAE"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        return torch.mean(torch.abs(y - yhat))


class Accuracy(Metric):
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        return (yhat.argmax(dim=1) == y).sum() / len(yhat)
