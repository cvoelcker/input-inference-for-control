## pendulum
python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name PendulumDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 4 --init-policy-variance 10000 --log-dir mixture_pendulum_s
python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name PendulumDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 4 --init-policy-variance 10000 --log-dir mixture_pendulum_l

## pendulum
python scripts/run_particle_i2c.py --num-particles 10 --num-policy-samples 5 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchPendulumDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --log-dir vsmc_pendulum_linear_policy
python scripts/run_particle_i2c.py --num-particles 10 --num-policy-samples 5 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchPendulumDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --log-dir vsmc_pendulum_mlp_policy
