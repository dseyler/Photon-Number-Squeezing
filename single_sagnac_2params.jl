
using MPI
MPI.Init()
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

N_t = 400;
dt = T_fiber / N_t;
t_grid = collect(range(0,T_fiber,length=N_t));

sim_fbs = sim(2,2^8,z_grid);

power_list = [0.01:0.025:3;]*1e8;
t_list = [0.01:0.0025:0.3;]

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

split_power_lists = approx_split(power_list, nprocs)
list_lengths = Array{Int64}(undef, nprocs)

for i in 1:nprocs
    	list_lengths[i] = length(split_power_lists[i])
end

my_power_list = split_power_lists[rank+1];

n_loops = length(my_power_list) * length(t_list)

# -
if rank == 0
	println(length(t_list) *length(power_list))
end

myphotons_in = zeros(length(my_power_list), length(t_list));
myphotons_out = zeros(length(my_power_list), length(t_list), sim_fbs.num_modes);
myphotons_fluc = 1.0im*zeros(length(my_power_list), length(t_list), sim_fbs.num_modes);

V_vac = vacuum_V(sim_fbs);

bs1 = beamsplitter([1,2],sqrt(0.1));
fiber1 = fiber(1,fiber_index,L_fiber,fiber_dispersion,fiber_nonlinearity);
fiber2 = fiber(2,fiber_index,L_fiber,fiber_dispersion,fiber_nonlinearity);
bs2 = beamsplitter([1,2],sqrt(0.1));

components = [bs1 fiber1 fiber2 bs2];

# +
for ii = 1:length(my_power_list)
	for jj = 1:length(t_list)
	     
		local bs1 = beamsplitter([1,2],sqrt(t_list[jj]));
        local fiber1 = fiber(1,fiber_index,L_fiber,fiber_dispersion,fiber_nonlinearity);
        local fiber2 = fiber(2,fiber_index,L_fiber,fiber_dispersion,fiber_nonlinearity);
        local bs2 = beamsplitter([1,2],sqrt(t_list[jj]));
        local components = [bs1 fiber1 fiber2 bs2];
        
		# Initializing the mean fields of the initial state
		
		center_amplitude = sqrt(my_power_list[ii]);
        state_sagnac = state(1.0im*zeros((sim_fbs.num_modes)*(sim_fbs.N_z)),V_vac);
        range_mode1 = get_row_index(1,fiber1.fiber_mode):get_row_index(N_z,fiber1.fiber_mode);
        state_sagnac.mean_fields[range_mode1] .= center_amplitude * sech.(sim_fbs.z_grid/L_pulse);
        myphotons_in[ii] = sum(abs2.(get_meanfield_i(state_sagnac,1)));

		# Solving for mean-field and fluctuation dynamics
		prop_system(components,state_sagnac,sim_fbs,t_grid)
								
		for ss=1:sim_fbs.num_modes
			myphotons_out[ii,jj,ss] = sum(abs2.(get_meanfield_i(state_sagnac,ss)));
            myphotons_fluc[ii,jj,ss] = n2_exp(state_sagnac,ss,sim_fbs);
		end
		if rank == 0
			i_count = (ii-1)*length(t_list) + jj
			println("$i_count / $n_loops")
		end
	end
end

mydata = (myphotons_in, myphotons_out, myphotons_fluc)

if rank > 0
	#Send data to 0th rank
	println("$rank: Sending mydata $rank -> 0/n")
	MPI.send(mydata, 0, rank+nprocs, comm)

else # rank == 0
    
	alldata  = [(Matrix{Float64}(undef, 0,0), Array{Float64}(undef, 0,0,0), Array{ComplexF64}(undef, 0,0,0)) for _ in 1:nprocs]
    photons_in = zeros(0,length(t_list));
    photons_out = zeros(0,length(t_list),sim_fbs.num_modes);
    photons_fluc = 1.0im*zeros(0,length(t_list),sim_fbs.num_modes);
	
	alldata[1] = mydata

	#Receive data from each rank
	for i in 1:nprocs-1
		println("Receiving from $i")
		alldata[i+1],statrcv = MPI.recv(i, i+nprocs, comm)
	end

	for data in alldata
		global photons_in = vcat(photons_in, data[1])
		global photons_out = vcat(photons_out, data[2])	
		global photons_fluc = vcat(photons_fluc, data[3])
	end

	if !isdir("outputs")
		mkdir("outputs")
	end

	npzwrite( "outputs/2param_photons_in_batch.npz",  photons_in)
	npzwrite( "outputs/2param_photons_out_batch.npz",  photons_out)
	npzwrite( "outputs/2param_photons_fluc_batch.npz",  photons_fluc)
	println("Done!")
end

MPI.Finalize()
