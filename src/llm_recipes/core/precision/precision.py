import torch
from contextlib import contextmanager


@contextmanager
def preserve_fp32_buffers(model: torch.nn.Module):
    fp32_buffers = dict()
    for name, param in model.named_buffers():
        if param.dtype == torch.float32:
            fp32_buffers[name] = param.clone()

    # model.to(torch.float16) or model.to(torch.bfloat16)
    yield

    for name, param in model.named_buffers():
        if name in fp32_buffers:
            if "." in name:
                module_name, buffer_name = name.rsplit(".", 1)
                target_module = model.get_submodule(module_name)
            else:
                buffer_name = name
                target_module = model
            setattr(target_module, buffer_name, fp32_buffers[name])
