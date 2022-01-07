#!/usr/bin/env python-real


import os
import sys
import subprocess
import sre_compile

def main(surf, out, rot, res, model, scal):
  fileDir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(fileDir)
  os.chdir('../../py/FiboSeg')


  command = f'/tools/anaconda3/envs/monai/bin/python predict_v3.py --surf {surf} --out {out} --rot {rot} --res {res} --model {model} --scal {scal}'
  print(command)
  os.system(command)


  #venv_python = "/tools/anaconda3/envs/monai/bin/python"
  #args = [venv_python,'predict_v2.py']
  #subprocess.run(args)
  

if __name__ == "__main__":
  if len (sys.argv) < 7:
    print("Usage: predictioncli <surf> <out> <rot> <res> <model> <scal>")
    sys.exit (1)
  main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],sys.argv[6])
