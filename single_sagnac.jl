include("pulsed_squeezed_fiber_lasers.jl")

λ_center = 1560e-9;
ω_center = 2*pi*c_light / λ_center;
fiber_index = 1.47;
v_group = c_light / fiber_index;
fiber_dispersion = 1*22e3 * 1e-30 * v_group^3; # in m^2/s
γ_fiber = 1.8*1e-3; # in 1/W/m.
fiber_nonlinearity = ħ*(ω_center)*(v_group^2)*(γ_fiber);
L_fiber = 10; # in meters
T_fiber = L_fiber / v_group;

# Pulse parameters - pulse time only makes sense for single-parameter pulses like sech / gaussian
t_pulse = 100e-15; # in seconds
L_pulse = v_group * t_pulse;

L_sim = 50*L_pulse;
N_z = 2^8;
z_grid = collect(range(-L_sim/2,L_sim/2,length=N_z));

N_t = 200;
dt = T_fiber / N_t;
t_grid = collect(range(0,T_fiber,length=N_t));

sim_fbs = sim(2,2^8,z_grid);

power_list = [0.01:0.25:2;]*1e4;
println(length(power_list))

photons_in = zeros(length(power_list));
photons_out = zeros(length(power_list),sim_fbs.num_modes);
photons_fluc = 1.0im*zeros(length(power_list),sim_fbs.num_modes);

V_vac = vacuum_V(sim_fbs);

bs1 = beamsplitter([1,2],sqrt(0.1));
fiber1 = fiber(1,fiber_index,L_fiber,fiber_dispersion,fiber_nonlinearity);
fiber2 = fiber(2,fiber_index,L_fiber,fiber_dispersion,fiber_nonlinearity);
bs2 = beamsplitter([1,2],sqrt(0.1));

components = [bs1 fiber1 fiber2 bs2];

function power_loop(ii)
	          
	# Initializing the mean fields of the initial state

	center_amplitude = power_list[ii];
	state_sagnac = state(1.0im*zeros((sim_fbs.num_modes)*(sim_fbs.N_z)),V_vac);
	range_mode1 = get_row_index(1,fiber1.fiber_mode):get_row_index(N_z,fiber1.fiber_mode);
	state_sagnac.mean_fields[range_mode1] .= center_amplitude * sech.(sim_fbs.z_grid/L_pulse);
	photons_in[ii] = sum(abs2.(get_meanfield_i(state_sagnac,1)));

	# Solving for mean-field and fluctuation dynamics
	prop_system(components,state_sagnac,sim_fbs,t_grid)
					          
	for ss=1:sim_fbs.num_modes
		photons_out[ii,ss] = sum(abs2.(get_meanfield_i(state_sagnac,ss)));
		photons_fluc[ii,ss] = n2_exp(state_sagnac,ss,sim_fbs);
	end
	println("(thread $(Threads.threadid()) of $(Threads.nthreads()))")
	println(ii)
end

Folds.map(power_loop, 1:length(power_list))

writedlm( "outputs/Test_photons_in.csv",  photons_in, ',')
writedlm( "outputs/Test_photons_out.csv",  photons_out, ',')
writedlm( "outputs/Test_photons_fluc.csv",  photons_fluc, ',')
