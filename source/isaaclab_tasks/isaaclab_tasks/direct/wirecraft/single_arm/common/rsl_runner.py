# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL runner that turns the env's per-step episode counts into a per-iteration success rate."""

from __future__ import annotations

from rsl_rl.runners import OnPolicyRunner

# Per-step counts the env logs under extras["log"]; summed here over the iteration, never averaged.
ENDED, SUCCEEDED = "counts/episodes_ended", "counts/episodes_succeeded"


class WireCraftOnPolicyRunner(OnPolicyRunner):
    """``OnPolicyRunner`` that logs ``metrics/success_rate`` as successes / episodes ended in the iteration.

    RSL-RL averages each logged key over the iteration's steps, so a rate logged per step would be an average of
    per-step ratios, and NaN in an iteration where no episode ended. Here the counts are summed instead:
    ``metrics/episodes_ended`` and ``metrics/episodes_succeeded`` are always logged, ``metrics/success_rate``
    only when at least one episode ended.
    """

    def log(self, locs: dict, *args, **kwargs):
        ended = succeeded = 0.0
        for info in locs.get("ep_infos", []):
            ended += float(info.pop(ENDED, 0.0))
            succeeded += float(info.pop(SUCCEEDED, 0.0))
        super().log(locs, *args, **kwargs)
        it = locs["it"]
        self.writer.add_scalar("metrics/episodes_ended", ended, it)
        self.writer.add_scalar("metrics/episodes_succeeded", succeeded, it)
        if ended > 0:
            self.writer.add_scalar("metrics/success_rate", succeeded / ended, it)
        rate = f"{succeeded / ended:.4f}" if ended > 0 else "n/a (no episode ended)"
        print(f"{'metrics/success_rate:':>35} {rate}  ({int(succeeded)}/{int(ended)} episodes)")
