import sys, os, setproctitle

if len(sys.argv) < 3:
    print("Usage: python3 run_exp.py <expname> <target_script> [args...]")
    sys.exit(1)

expname = sys.argv[1]
target_script = sys.argv[2]
script_args = sys.argv[3:]

setproctitle.setproctitle(f"python train_gen.py exp={expname}")
sys.argv = [target_script] + script_args
exec(compile(open(target_script).read(), target_script, 'exec'))
