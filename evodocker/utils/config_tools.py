import importlib.util

import ml_collections as mlc


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def enforce_config_constraints(config):
    def string_to_setting(s):
        path = s.split('.')
        setting = config
        for p in path:
            setting = setting.get(p)

        return setting

    mutually_exclusive_bools = [
        (
            "globals.use_lma",
            "globals.use_flash",
            "globals.use_deepspeed_evo_attention"
        ),
    ]

    for options in mutually_exclusive_bools:
        option_settings = [string_to_setting(o) for o in options]
        if sum(option_settings) > 1:
            raise ValueError(f"Only one of {', '.join(options)} may be set at a time")

    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    if config.globals.use_flash and not fa_is_installed:
        raise ValueError("use_flash requires that FlashAttention is installed")

    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    ds4s_is_installed = deepspeed_is_installed and importlib.util.find_spec(
        "deepspeed.ops.deepspeed4science") is not None
    if config.globals.use_deepspeed_evo_attention and not ds4s_is_installed:
        raise ValueError(
            "use_deepspeed_evo_attention requires that DeepSpeed be installed "
            "and that the deepspeed.ops.deepspeed4science package exists"
        )