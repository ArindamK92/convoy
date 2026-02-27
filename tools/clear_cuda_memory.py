"""Small helper to clear CUDA memory after a run."""


def clear_cuda_memory(model=None, trainer=None, env=None, batch=None, out=None):
    """Drop common references, run GC, and clear CUDA allocator caches."""
    del model, trainer, env, batch, out
    import gc, torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":
    clear_cuda_memory()
