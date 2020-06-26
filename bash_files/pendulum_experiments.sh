## pendulum
# python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name PendulumKnown --type mixture --smoothing smoothing --init-alpha 0.01 --components 4 --init-policy-variance 1 --log-dir mixture_pendulum_s --init-state-var 0.01 --Q 100 1 --init-state-mean 0 -1 0 --shape pendulum --u-clipping 2.
# python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name PendulumKnown --type mixture --smoothing smoothing --init-alpha 0.01 --components 4 --init-policy-variance 1 --log-dir mixture_pendulum_l --init-state-var 0.01 --Q 100 1 --init-state-mean 0 -1 0 --shape pendulum --u-clipping 2.
python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 20 --batch-size 16 --horizon 100 --log-every 4 --em-steps 100 --env-name PendulumKnown --type mixture --smoothing smoothing --init-alpha 0.01 --components 2 --init-policy-variance 2 --log-dir mixture_pendulum_s_exploring --init-state-var 0.01 --Q 100 1 --init-state-mean 0 -1 0 --shape pendulum --u-clipping 4. --clip-alpha
## pendulum
python scripts/run_particle_i2c.py --num-particles 10 --num-policy-samples 5 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250000 --env-name TorchPendulumKnown --type VSMC --smoothing greedy --init-alpha 0.01 --components 1 --log-dir vsmc_pendulum_linear_policy --init-state-var 0.01 --Q 100 1 --init-state-mean 0 -1 0 --shape pendulum --u-clipping 2.
python scripts/run_particle_i2c.py --num-particles 10 --num-policy-samples 5 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250000 --env-name TorchPendulumKnown --type VSMC --smoothing greedy --init-alpha 0.01 --components 1 --log-dir vsmc_pendulum_mlp_policy --init-state-var 0.01 --Q 100 1 --init-state-mean 0 -1 0 --shape pendulum --u-clipping 2.
