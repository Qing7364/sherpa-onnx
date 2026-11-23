#!/usr/bin/env python3
"""Create public/synthetic decoder fixtures using the tested input builder."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

from test_decoder_logits_mode import make_inputs
from compare_decoder_logits import options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))["thinker_config"]
    text = config["text_config"]
    descriptor = SimpleNamespace(audio_token_id=config["audio_token_id"],
        hidden_size=text["hidden_size"], num_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"], num_layers=text["num_hidden_layers"])
    args.output.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.decoder), sess_options=options(4),
                                   providers=["CPUExecutionProvider"])
    names = [x.name for x in session.get_inputs()]
    output_names = [x.name for x in session.get_outputs()]

    def save(name, values, source):
        target = args.output / f"{name}.npz"
        if target.exists():
            raise ValueError("Refusing to replace frozen decoder inputs")
        feed = {key: value.numpy().copy() for key, value in zip(names, values)}
        feed["next_tokens"] = np.arange(100, 116, dtype=np.int64)
        np.savez_compressed(target, **feed)
        metadata = dict(source=source, seq=int(feed["input_ids"].shape[1]),
            past_len=int(feed["cache_position"][0]), seed=42,
            continuation="Force the same known synthetic token IDs 100..115",
            sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return feed

    for seq in (1, 8, 64, 128, 256, 448, 511):
        save(f"synthetic-s{seq:03d}-past000", make_inputs(descriptor, seq), "Test input builder, fixed seed")
    prefix = {key: value.numpy() for key, value in zip(names, make_inputs(descriptor, 64))}
    deltas = dict(zip(output_names, session.run(None, prefix)))
    continuation = make_inputs(descriptor, 8, past_len=64)
    for layer in range(descriptor.num_layers):
        for kind, slot in (("key", 0), ("value", 1)):
            continuation[4 + layer * 2 + slot][:, :64] = torch.from_numpy(deltas[f"{kind}_delta_{layer}"])
    save("synthetic-s008-past064", continuation,
         "Legal continuation from actual all-model S=64 prefix KV deltas")
    print("Saved 8 fixed decoder inputs; S=511 permits only one next token")


if __name__ == "__main__":
    main()
