"""ROOT adapter.

PyROOT:   from allegro_style import root as aroot; R = aroot.use()
C++:      gROOT->LoadMacro("<path>/allegro_style/root/AllegroStyle.C"); SetAllegroStyle();

After editing style.json run `python -m allegro_style.root` to regenerate
AllegroStyleConfig.h (the tests fail if it is out of date).
"""
import json
from pathlib import Path

from . import load_config

ROOT_DIR = Path(__file__).with_name("root")
MACRO = ROOT_DIR / "AllegroStyle.C"
HEADER = ROOT_DIR / "AllegroStyleConfig.h"


def _cpp_type(v):
    if isinstance(v, bool):
        return "const bool"
    if isinstance(v, int):
        return "const int"
    if isinstance(v, float):
        return "const double"
    return "const char* const"


def _cpp_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    return repr(v)


def _flatten(d, prefix=""):
    for k, v in d.items():
        name = prefix + k
        if isinstance(v, dict):
            yield from _flatten(v, name + "_")
        elif isinstance(v, list):
            # a numeric list with any float becomes double, e.g. [1, 0.75]
            is_float = any(isinstance(x, float) for x in v)
            items = ", ".join(_cpp_value(float(x) if is_float else x) for x in v)
            yield f"{_cpp_type(0.0 if is_float else v[0])} {name}[] = {{{items}}};"
            yield f"const int {name}_n = {len(v)};"
        else:
            yield f"{_cpp_type(v)} {name} = {_cpp_value(v)};"


def header_text(cfg=None):
    lines = "\n".join(_flatten(cfg or load_config()))
    return (
        "// Generated from style.json by `python -m allegro_style.root`. Do not edit.\n"
        "#ifndef ALLEGRO_STYLE_CONFIG_H\n#define ALLEGRO_STYLE_CONFIG_H\n"
        f"namespace AllegroCfg {{\n{lines}\n}}\n#endif\n"
    )


def write_header(path=HEADER):
    Path(path).write_text(header_text())
    return path


def load():
    """Load AllegroStyle.C into ROOT and return the ROOT module."""
    import ROOT
    if not hasattr(ROOT, "SetAllegroStyle"):
        ROOT.gROOT.LoadMacro(str(MACRO))
    return ROOT


def use():
    """Apply the ALLEGRO style to ROOT and return the ROOT module."""
    R = load()
    R.SetAllegroStyle()
    return R


if __name__ == "__main__":
    print(write_header())
