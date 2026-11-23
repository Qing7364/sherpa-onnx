#!/usr/bin/env python3
"""Build local-only CLI copies exposing language hints and generated-token counts.

No production source is changed. Reuse the exact Release compiler/linker flags.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Instrumentation anchor is not unique: {before}")
    return text.replace(before, after, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build = source / "build-prefill"
    commands = json.loads((build / "compile_commands.json").read_text())
    recorded = []

    def compile_copy(original_name, copy, object_path):
        entry, = [e for e in commands if Path(e["file"]).name == original_name]
        command = shlex.split(entry["command"])
        command[command.index("-o") + 1] = str(object_path)
        command[command.index(entry["file"])] = str(copy)
        recorded.append(dict(cwd=entry["directory"], command=command))
        subprocess.run(command, cwd=entry["directory"], check=True)

    cli_name = "sherpa-onnx-offline.cc"
    cli = (source / "sherpa-onnx/csrc" / cli_name).read_text()
    cli = replace_once(cli, "  std::string length_reward;", """  std::string bench_language;
  po.Register("bench-language", &bench_language, "Local test-only language hint");
  std::string length_reward;""")
    cli = replace_once(cli, "    auto s = recognizer.CreateStream();", """    auto s = recognizer.CreateStream();
    if (!bench_language.empty()) {
      s->SetOption("language", bench_language);
    }""")
    cli_copy = output / cli_name
    cli_copy.write_text(cli)
    cli_obj = output / (cli_name + ".o")
    compile_copy(cli_name, cli_copy, cli_obj)
    impl_name = "offline-recognizer-qwen3-asr-impl.cc"
    impl_path = "sherpa-onnx/csrc/" + impl_name
    old = subprocess.run(["git", "show", "8a667ab114e8a007f671678ddb031729dd81dce4:" + impl_path],
                         cwd=source, check=True, capture_output=True, text=True).stdout
    current = (source / impl_path).read_text()
    for label, implementation in (("baseline", old), ("candidate", current)):
        implementation = replace_once(implementation,
            "  std::vector<int64_t> cleaned_ids = generated_ids;", """  SHERPA_ONNX_LOGE("QWEN_BENCH_GENERATED=%zu PREFILL=%d FINAL_CONTEXT=%d EOS=%d",
                   generated_ids.size(), context_len, cur_len, next_id == eos_id);
  std::vector<int64_t> cleaned_ids = generated_ids;""")
        impl_copy = output / (label + "-" + impl_name)
        impl_copy.write_text(implementation)
        impl_obj = output / (label + "-impl.o")
        compile_copy(impl_name, impl_copy, impl_obj)
        link = shlex.split((build / "sherpa-onnx/csrc/CMakeFiles/sherpa-onnx-offline.dir/link.txt").read_text())
        link[link.index("CMakeFiles/sherpa-onnx-offline.dir/sherpa-onnx-offline.cc.o")] = str(cli_obj)
        link.insert(link.index("-o"), str(impl_obj))
        link[link.index("-o") + 1] = str(output / (label + "-sherpa-onnx-offline"))
        recorded.append(dict(cwd=str(build / "sherpa-onnx/csrc"), command=link))
        subprocess.run(link, cwd=build / "sherpa-onnx/csrc", check=True)
    (output / "build-commands.json").write_text(json.dumps(recorded, indent=2) + "\n")


if __name__ == "__main__":
    main()
