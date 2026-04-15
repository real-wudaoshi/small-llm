import config
from model import SmallLlm
from torchinfo import summary


def _safe_storage_key(param):
    try:
        return ("storage", param.untyped_storage().data_ptr(), tuple(param.shape), str(param.dtype))
    except Exception:
        return ("object", id(param))


def count_params_recursive_dedup(model):
    total = 0
    trainable = 0
    seen_obj = set()
    seen_storage = set()
    duplicate_bindings = 0

    for _, module in model.named_modules(remove_duplicate=False):
        for _, param in module.named_parameters(recurse=False):
            obj_key = id(param)
            storage_key = _safe_storage_key(param)
            if obj_key in seen_obj or storage_key in seen_storage:
                duplicate_bindings += 1
                continue
            seen_obj.add(obj_key)
            seen_storage.add(storage_key)
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
    return total, trainable, duplicate_bindings


def count_params_recursive_raw(model):
    total = 0
    trainable = 0
    for _, module in model.named_modules(remove_duplicate=False):
        for _, param in module.named_parameters(recurse=False):
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
    return total, trainable


def fmt(n):
    return f"{n:,}"


def main():
    model = SmallLlm(
        config.MAX_VOCAB_SIZE,
        config.DIM,
        config.FFN_DIM,
        config.HEADS,
        config.LAYERS,
    )

    raw_total, raw_trainable = count_params_recursive_raw(model)
    dedup_total, dedup_trainable, dup_bindings = count_params_recursive_dedup(model)

    print("=== Recursive Parameter Count ===")
    print(f"raw_total                : {fmt(raw_total)}")
    print(f"raw_trainable            : {fmt(raw_trainable)}")
    print(f"dedup_total              : {fmt(dedup_total)}")
    print(f"dedup_trainable          : {fmt(dedup_trainable)}")
    print(f"duplicate_binding_refs   : {fmt(dup_bindings)}")
    print(f"dedup_reduction          : {fmt(raw_total - dedup_total)}")
    print("")

    summary(model)


if __name__ == "__main__":
    main()