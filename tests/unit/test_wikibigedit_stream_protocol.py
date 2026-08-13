import json
import subprocess
import sys

from examples.wikibigedit_stream_protocol import build_protocol, cohort_indices


def test_cohorts_are_deterministic_bounded_and_spread():
    first=cohort_indices(1000,100,42)
    assert first==cohort_indices(1000,100,42)
    assert len(first)==100
    assert min(first)>=0 and max(first)<1000
    assert max(first)-min(first)>900


def test_retention_matrix_only_uses_past_cohorts():
    protocol=build_protocol([{}]*10_000,(1000,5000,10000),50,42)
    assert set(protocol["retention"]["1000"])=={"1000"}
    assert set(protocol["retention"]["10000"])=={"1000","5000","10000"}


def test_script_entrypoint_runs_from_outside_repo(tmp_path):
    manifest=tmp_path/"manifest.jsonl"; output=tmp_path/"protocol.json"
    manifest.write_text("\n".join(json.dumps({"case_id":str(i)}) for i in range(4)))
    script=__import__("examples.wikibigedit_stream_protocol",fromlist=["x"]).__file__
    subprocess.run([sys.executable,script,"--manifest",str(manifest),"--output",str(output),"--points","2","4"],cwd=tmp_path,check=True,capture_output=True,text=True)
    assert json.loads(output.read_text())["points"]==[2,4]
