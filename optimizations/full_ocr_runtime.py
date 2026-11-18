"""Windows runner for the full-data experiment. Process the final loader batch."""
import inspect
import runpy
import sys
from pathlib import Path

from ocr_runtime import install_ctc_bridge

script = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(script.parent.parent))
sys.modules["torch"] = None
install_ctc_bridge()

from tools import program

source = inspect.getsource(program.train)
assert source.count("len(train_dataloader) - 1") == 3
# Keep the third occurrence (the logging condition) unchanged.
source = source.replace("len(train_dataloader) - 1", "len(train_dataloader)", 2)
exec(compile(source, program.__file__, "exec"), vars(program))

if sys.argv.pop(1) == "formula":
    # Reuse the encoder-freezing behavior validated in the fraction pilot.
    sys.argv.insert(1, str(script))
    runpy.run_path(str(Path(__file__).with_name("formula_runtime.py")), run_name="__main__")
else:
    runpy.run_path(str(script), run_name="__main__")
