"""Run the actual launcher with only executable resolution redirected to a fake child."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'run.py'
BOOTSTRAP='''
import os, runpy, subprocess, sys
original_run = subprocess.run
def fake_executable(cmd, **kwargs):
    assert cmd[0] == 'vllm'
    if os.environ.get('FAKE_MISSING') == '1':
        raise FileNotFoundError('fake vllm unavailable')
    return original_run([sys.executable, os.environ['FAKE_CHILD'], *cmd[1:]], **kwargs)
subprocess.run = fake_executable
script=sys.argv[1]
sys.argv=sys.argv[1:]
runpy.run_path(script, run_name='__main__')
'''
CHILD='''
import os, json, sys
from pathlib import Path
Path(os.environ['FAKE_RECORD']).write_text(json.dumps({'args':sys.argv[1:], 'env':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','VLLM_LOGGING_LEVEL']}}))
sys.exit(int(os.environ['FAKE_CODE']))
'''

class ExitTests(unittest.TestCase):
    def invoke(self, code, missing=False):
        with tempfile.TemporaryDirectory() as tmp:
            fake=Path(tmp)/'fake_vllm.py';fake.write_text(CHILD)
            record=Path(tmp)/'record.json'
            env={**os.environ,'FAKE_CHILD':str(fake),'FAKE_RECORD':str(record),
                 'FAKE_CODE':str(code),'FAKE_MISSING':str(int(missing))}
            p=subprocess.run([sys.executable,'-c',BOOTSTRAP,str(SCRIPT),'qwen-3',
                              '--gpu-id','2','3','--port','5012','--max-tokens','2048',
                              '--enable-tool','--log-debug','--model-path','local-checkpoint'],
                             env=env,capture_output=True,text=True)
            return p,json.loads(record.read_text()) if record.exists() else None

    def test_child_status_and_logs(self):
        for code in [0,7]:
            with self.subTest(code=code):
                p,record=self.invoke(code)
                self.assertEqual(p.returncode,code,p.stderr)
                self.assertIn('LLM server exited successfully' if code==0 else 'LLM server exited with code 7',p.stderr)
                self.assertEqual(record['args'],['serve','local-checkpoint','--served-model-name','Qwen/Qwen3-1.7B',
                    '--tensor-parallel-size','2','--dtype','bfloat16','--max-model-len','2048',
                    '--gpu-memory-utilization','0.85','--host','0.0.0.0','--port','5012',
                    '--max-num-seqs','64','--trust-remote-code','--enable-prefix-caching',
                    '--enable-expert-parallel','--enable-chunked-prefill','--enable-auto-tool-choice',
                    '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3'])
                self.assertEqual(record['env'],{'CUDA_VISIBLE_DEVICES':'2,3','CUDA_DEVICE_ORDER':'PCI_BUS_ID','VLLM_LOGGING_LEVEL':'DEBUG'})

    def test_startup_failure(self):
        p,record=self.invoke(0,missing=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('FileNotFoundError',p.stderr)
        self.assertIsNone(record)

if __name__=='__main__':unittest.main()
