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
        ),
    ]

    for options in mutually_exclusive_bools:
        option_settings = [string_to_setting(o) for o in options]
        if sum(option_settings) > 1:
            raise ValueError(f"Only one of {', '.join(options)} may be set at a time")
