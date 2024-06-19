# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def convert_deprecated_v1_keys(state_dict):
    """Update older OpenFold model weight names to match the current model code."""

    replacements = {
        'core.msa_transition': 'msa_transition',
        'core.tri_': 'pair_stack.tri_',
        'core.pair_transition': 'pair_stack.pair_transition',
        'ipa.linear_q_points': 'ipa.linear_q_points.linear',
        'ipa.linear_kv_points': 'ipa.linear_kv_points.linear'
    }

    convert_key_re = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))

    converted_state_dict = {}
    for key, value in state_dict.items():
        # For each match, look-up replacement value in the dictionary
        new_key = convert_key_re.sub(lambda m: replacements[m.group()], key)

        converted_state_dict[new_key] = value

    return converted_state_dict


def import_openfold_weights_(model, state_dict):
    """
    Import model weights. Several parts of the model were refactored in the process
    of adding support for Multimer. The state dicts of older models are translated
    to match the refactored model code.
    """
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        converted_state_dict = convert_deprecated_v1_keys(state_dict)
        model.load_state_dict(converted_state_dict)
