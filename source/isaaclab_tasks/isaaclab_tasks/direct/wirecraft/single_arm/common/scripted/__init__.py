# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted policies. They act only through the env's normal action interface."""

from __future__ import annotations


def make_policy(env):
    from .channel import ChannelPolicy
    from .insertion import InsertionPolicy
    from .routing import RoutingPolicy

    return {"insertion": InsertionPolicy, "clip_routing": RoutingPolicy, "lay_in_channel": ChannelPolicy}[
        env.cfg.task.name
    ](env)
