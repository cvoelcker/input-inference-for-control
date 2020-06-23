#### MIXTURE EVALUATIONS

## linear
python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 1 --init-policy-variance 10000 --log-dir mixture_linear_s
python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 2 --init-policy-variance 10000 --log-dir mixture_bimodal_s --init-state-bimodal

## linear large number particles
python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 1 --init-policy-variance 10000 --log-dir mixture_linear_l
python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 2 --init-policy-variance 10000 --log-dir mixture_bimodal_l

## linear high uncertainty
python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 1 --init-policy-variance 10000 --init-state-var 1. --log-dir mixture_linear_large_var

# ## pendulum
# python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name PendulumDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 4 --init-policy-variance 10000 --log-dir mixture_pendulum_s
# python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name PendulumDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 4 --init-policy-variance 10000 --log-dir mixture_pendulum_l


#### VSMC Evaluations

## linear
python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchLinearDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --log-dir vsmc_linear

## bimodal
python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchLinearDisturbed --type VSMC --smoothing greedy-init-alpha 0.00001 --components 1 --log-dir vsmc_bimodal_linear_policy
python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchLinearDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --nn-type LogMlp --log-dir vsmc_bimodal_mlp_policy

# ## pendulum
# python scripts/run_particle_i2c.py --num-particles 10 --num-policy-samples 5 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchPendulumDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --log-dir vsmc_pendulum_linear_policy
# python scripts/run_particle_i2c.py --num-particles 10 --num-policy-samples 5 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchPendulumDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --log-dir vsmc_pendulum_mlp_policy

