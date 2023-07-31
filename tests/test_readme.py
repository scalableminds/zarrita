from pathlib import Path
from shutil import rmtree
from typing import Callable, Dict


def test_readme() -> None:
    path = Path("testoutput")
    rmtree(path, ignore_errors=True)

    readme_text = (Path(__file__).parent.parent / "README.md").read_text()

    code_blocks = []
    is_code_block = False
    for line in readme_text.splitlines():
        if line.startswith("```"):
            is_code_block = not is_code_block
            if is_code_block:
                code_blocks.append("")
            continue
        if is_code_block:
            code_blocks[-1] += line + "\n"

    new_code = ""
    for i, code_block in enumerate(code_blocks):
        if "await" in code_block:
            new_code += (
                f"async def async_fn{i}():\n"
                + "\n".join("    " + line for line in code_block.splitlines())
                + "\n"
                + f"sync(async_fn{i}())"
                + "\n"
            )
        else:
            new_code += code_block + "\n"

    new_code = (
        "def main():\n"
        + "    from zarrita.sync import sync\n"
        + "\n".join("    " + line for line in new_code.splitlines())
    )
    fake_module_dict: Dict[str, Callable[[], None]] = {}
    exec(new_code, fake_module_dict)
    fake_module_dict["main"]()
