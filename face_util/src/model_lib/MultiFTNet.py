# MultiFTNetInfer.py
import torch
from torch import nn
from face_util.src.model_lib.MiniFASNet import MiniFASNetV2SE


class MultiFTNetInfer(nn.Module):
    """
    Inference-only:
      - uses MiniFASNet backbone
      - no FTGenerator
      - always returns cls logits
    """
    def __init__(self, img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5)):
        super().__init__()
        self.model = MiniFASNetV2SE(
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            num_classes=num_classes,
            img_channel=img_channel,
        )

    def forward(self, x):
        # same forward as MultiFTNet but without FT branch
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x = self.model.conv_45(x)
        x = self.model.conv_5(x)
        x = self.model.conv_6_sep(x)
        x = self.model.conv_6_dw(x)
        x = self.model.conv_6_flatten(x)
        x = self.model.linear(x)
        x = self.model.bn(x)
        x = self.model.drop(x)   # dropout disabled automatically in eval()
        cls = self.model.prob(x)
        return cls


def load_infer_model(weights_path: str, *, num_threads: int | None = None) -> MultiFTNetInfer:
    """
    Loads weights and sets optimal inference settings for CPU.
    """
    if num_threads:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, min(4, num_threads // 2)))

    m = MultiFTNetInfer()
    sd = torch.load(weights_path, map_location="cpu")
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m
