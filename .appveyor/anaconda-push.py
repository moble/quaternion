import os
import glob
import subprocess
import traceback

token = os.environ['ANACONDA_TOKEN']
cmd = ['anaconda', '--token', token, 'upload', '--no-progress', '--force']
cmd.extend(glob.glob('*.tar.bz2'))
try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError:
    traceback.print_exc()
