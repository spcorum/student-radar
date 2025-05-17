import os
import sys


model  = sys.argv[1]
epoch = sys.argv[2]
name = sys.argv[3]

os.system(f'python src/generate/save_conditional_generations.py {model} {epoch} {name}')

os.makedirs("plots/generation", exist_ok=True)
os.system(f'mkdir plots/generation/{name}')

os.system(f'python src/postprocessing/rota2030-bsm-rsp/main2.py {name}')
