"""Paddle-only runtime for the Windows OCR trainer."""
import runpy
import sys
from pathlib import Path


def install_ctc_bridge():
    import paddle
    original = paddle.nn.functional.ctc_loss

    def portable_ctc(log_probs, labels, input_lengths, label_lengths, *args, **kwargs):
        if not log_probs.place.is_gpu_place():
            return original(log_probs, labels, input_lengths, label_lengths, *args, **kwargs)
        cpu_labels = labels.cpu()
        cpu_input_lengths = input_lengths.cpu()
        cpu_label_lengths = label_lengths.cpu()

        class CPUCTC(paddle.autograd.PyLayer):
            @staticmethod
            def forward(ctx, logits):
                ctx.place = logits.place
                ctx.save_for_backward(logits.detach().cpu())
                loss = original(logits.cpu(), cpu_labels, cpu_input_lengths,
                                cpu_label_lengths, *args, **kwargs)
                return loss.to(logits.place)

            @staticmethod
            def backward(ctx, grad):
                (logits,) = ctx.saved_tensor()
                with paddle.enable_grad():
                    logits.stop_gradient = False
                    loss = original(logits, cpu_labels, cpu_input_lengths,
                                    cpu_label_lengths, *args, **kwargs)
                    (gradient,) = paddle.grad([loss], [logits], grad_outputs=[grad.cpu()])
                return gradient.to(ctx.place)

        return CPUCTC.apply(log_probs)

    paddle.nn.functional.ctc_loss = portable_ctc


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    # The training pipeline only uses Paddle. Albumentations optionally loads
    # PyTorch and conflicts with Paddle's CUDA DLLs in this Windows environment.
    sys.modules["torch"] = None
    if sys.platform == "win32":
        install_ctc_bridge()
    script = Path(sys.argv.pop(1))
    if script.name == "export_model.py":
        from paddle.jit.dy2static import utils
        directory = Path(sys.argv[sys.argv.index("-c") + 1]).parent / "export_tmp"
        directory.mkdir(exist_ok=True)
        utils.get_temp_dir = lambda: str(directory)
    runpy.run_path(str(script), run_name="__main__")
