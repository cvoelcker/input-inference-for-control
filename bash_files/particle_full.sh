#### MIXTURE EVALUATIONS

## linear
python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.0001 --components 1 --init-policy-variance 10000 --log-dir mixture_linear_s --exp-factor 2.
# python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 2 --init-policy-variance 10000 --log-dir mixture_bimodal_s --init-state-bimodal --init-state-var 0.25
# 
# # linear large number particles
# python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 1 --init-policy-variance 10000 --log-dir mixture_linear_l --init-state-var 0.25
# python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 2 --init-policy-variance 10000 --log-dir mixture_bimodal_l --init-state-bimodal --init-state-var 0.25
# 
# ## linear high uncertainty
# python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 16 --horizon 100 --log-every 1 --em-steps 100 --env-name LinearDisturbed --type mixture --smoothing smoothing --init-alpha 0.00001 --components 1 --init-policy-variance 10000 --init-state-var 2. --log-dir mixture_linear_large_var
# 
# 
# #### VSMC Evaluations
# 
# ## linear
# python scripts/run_particle_i2c.py --num-particles 100 --num-policy-samples 10 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchLinearDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --log-dir vsmc_linear --init-state-var 0.25
# 
# ## bimodal
# python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchLinearDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --log-dir vsmc_bimodal_linear_policy --init-state-var 0.25
# python scripts/run_particle_i2c.py --num-particles 500 --num-policy-samples 50 --batch-size 1 --horizon 100 --log-every 1000 --em-steps 250 --env-name TorchLinearDisturbed --type VSMC --smoothing greedy --init-alpha 0.00001 --components 1 --nn-type LogMlp --log-dir vsmc_bimodal_mlp_policy --init-state-var 0.25
# 
