"""Serve an experimental tactile-guided JAX Sharpa policy; no actuation."""

import argparse
from pathlib import Path

from openpi.guidance.tactile import GuidedPolicy
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=0.4)
    parser.add_argument("--step-rms", type=float, default=0.03)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    base = policy_config.create_trained_policy(
        config.get_config("sharpa_egg"),
        args.checkpoint,
        default_prompt="pick up the egg",
        sample_kwargs={"num_steps": args.num_steps},
    )
    guided = GuidedPolicy(
        base, args.guide, scale=args.scale, max_time=args.max_time, max_step_rms=args.step_rms, num_steps=args.num_steps
    )
    WebsocketPolicyServer(policy=guided, host=args.host, port=args.port, metadata=guided.metadata).serve_forever()


if __name__ == "__main__":
    main()
