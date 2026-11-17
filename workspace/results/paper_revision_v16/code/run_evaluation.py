"""Run the frozen evaluation suites sequentially; never co-time numerical jobs."""
import subprocess,sys
from pathlib import Path
here=Path(__file__).parent
for suite in ('static','brugge','synthetic','dynamic'):
    print('\nStarting frozen suite:',suite,flush=True)
    subprocess.run([sys.executable,'-u',str(here/'benchmark.py'),suite,'--repeats','3','--reliable'],check=True)
subprocess.run([sys.executable,'-u',str(here/'certify_results.py')],check=True)
