function homochirality_study()
    % Study which channels can convert Hamiltonian evolution to homochiral state
    % H = (a/2)*σ_z + (b/2)*σ_x
    
    fprintf('=========================================\n');
    fprintf('HOMOCHIRALITY CHANNEL CONVERSION STUDY\n');
    fprintf('=========================================\n\n');
    
    % Parameters for the asymmetric double well
    a = 1.0;      % Energy asymmetry
    b = 0.0;      % Tunneling amplitude (set to 0 for no tunneling)
    t = 1.0;      % Evolution time
    
    fprintf('Double-well Hamiltonian parameters:\n');
    fprintf('  a = %.2f (energy asymmetry)\n', a);
    fprintf('  b = %.2f (tunneling)\n', b);
    fprintf('  t = %.2f (evolution time)\n', t);
    
    % Create the Hamiltonian
    H = create_double_well_hamiltonian(a, b);
    fprintf('\nHamiltonian H = (a/2)*σ_z + (b/2)*σ_x:\n');
    disp(H);
    
    % Eigenvalues and eigenvectors
    [V, D] = eig(H);
    fprintf('Eigenvalues: [%.4f, %.4f]\n', D(1,1), D(2,2));
    fprintf('Eigenstates (columns):\n');
    disp(V);
    
    % Create unitary channel from Hamiltonian evolution
    fprintf('\n--- Source Channel: Hamiltonian Evolution ---\n');
    param_H.H = H;
    param_H.t = t;
    J_Phi = create_channel('hamiltonian', 2, param_H);
    
    fprintf('Unitary U = exp(-iHt):\n');
    U = expm(-1i * H * t);
    disp(U);
    
    fprintf('J(Φ) - Choi matrix of Hamiltonian evolution:\n');
    fprintf('Size: %dx%d\n', size(J_Phi));
    
    % Target: Completely homochiral state
    % This means we want a channel that maps any state to |L⟩⟨L| or |R⟩⟨R|
    fprintf('\n--- Target Channels: Homochiral States ---\n');
    
    % Test 1: Channel that outputs |L⟩⟨L| (left homochiral)
    fprintf('\nTest 1: Target = Constant channel → |L⟩⟨L|\n');
    rho_L = [1, 0; 0, 0];  % |L⟩⟨L| = |0⟩⟨0|
    J_Psi_L = create_constant_channel(rho_L);
    
    result_L = solve_conversion_sdp(J_Phi, J_Psi_L, 'H → |L⟩⟨L|');
    
    % Test 2: Channel that outputs |R⟩⟨R| (right homochiral)
    fprintf('\nTest 2: Target = Constant channel → |R⟩⟨R|\n');
    rho_R = [0, 0; 0, 1];  % |R⟩⟨R| = |1⟩⟨1|
    J_Psi_R = create_constant_channel(rho_R);
    
    result_R = solve_conversion_sdp(J_Phi, J_Psi_R, 'H → |R⟩⟨R|');
    
    % Test 3: Channel that outputs equal superposition (achiral)
    fprintf('\nTest 3: Target = Constant channel → |+⟩⟨+|\n');
    rho_plus = 0.5 * [1, 1; 1, 1];  % |+⟩⟨+|
    J_Psi_plus = create_constant_channel(rho_plus);
    
    result_plus = solve_conversion_sdp(J_Phi, J_Psi_plus, 'H → |+⟩⟨+|');
    
    % Test 4: What about with tunneling?
    if b == 0
        fprintf('\n\n=========================================\n');
        fprintf('TESTING WITH TUNNELING (b ≠ 0)\n');
        fprintf('=========================================\n');
        
        b_tunnel = 0.5;
        H_tunnel = create_double_well_hamiltonian(a, b_tunnel);
        fprintf('\nWith tunneling b = %.2f:\n', b_tunnel);
        fprintf('Hamiltonian:\n');
        disp(H_tunnel);
        
        param_H_tunnel.H = H_tunnel;
        param_H_tunnel.t = t;
        J_Phi_tunnel = create_channel('hamiltonian', 2, param_H_tunnel);
        
        fprintf('\nTest 4: H(with tunneling) → |L⟩⟨L|\n');
        result_tunnel = solve_conversion_sdp(J_Phi_tunnel, J_Psi_L, 'H(tunneling) → |L⟩⟨L|');
    end
    
    % Summary
    fprintf('\n\n=========================================\n');
    fprintf('SUMMARY\n');
    fprintf('=========================================\n');
    fprintf('Conversion to |L⟩⟨L|: ');
    if result_L < 1e-6
        fprintf('POSSIBLE (distance = %.2e)\n', result_L);
    else
        fprintf('IMPOSSIBLE (distance = %.6f)\n', result_L);
    end
    
    fprintf('Conversion to |R⟩⟨R|: ');
    if result_R < 1e-6
        fprintf('POSSIBLE (distance = %.2e)\n', result_R);
    else
        fprintf('IMPOSSIBLE (distance = %.6f)\n', result_R);
    end
    
    fprintf('Conversion to |+⟩⟨+|: ');
    if result_plus < 1e-6
        fprintf('POSSIBLE (distance = %.2e)\n', result_plus);
    else
        fprintf('IMPOSSIBLE (distance = %.6f)\n', result_plus);
    end
end

function distance = solve_conversion_sdp(J_Phi, J_Psi, description)
    % Solve the channel conversion SDP
    
    dX = 2; dY = 2; dXp = 2; dYp = 2;
    
    cvx_begin sdp quiet
        dim_Theta = dYp * dXp * dY * dX;
        dim_J1 = dXp * dY * dX;
        
        variable J_Theta(dim_Theta, dim_Theta) hermitian semidefinite
        variable J1(dim_J1, dim_J1) hermitian semidefinite
        
        I_Yp = eye(dYp);
        I_Xp = eye(dXp);
        I_Y = eye(dY);
        I_X = eye(dX);
        
        % Constraint 1: Tr_Y' J(Θ) = J1
        sys_dims = [dYp, dXp, dY, dX];
        J_Theta_traced = PartialTrace(J_Theta, 1, sys_dims);
        J_Theta_traced == J1;
        
        % Constraint 2: Tr_Y J1 = I_X' ⊗ I_X
        J1_dims = [dXp, dY, dX];
        J1_traced = PartialTrace(J1, 2, J1_dims);
        J1_traced == kron(I_Xp, I_X);
        
        % Constraint 3: Channel conversion (with slack)
        J_Theta_pt = PartialTranspose(J_Theta, [3, 4], sys_dims);
        I_YpXp = kron(I_Yp, I_Xp);
        rhs_operator = kron(I_YpXp, J_Phi);
        product = J_Theta_pt * rhs_operator;
        Theta_Phi = PartialTrace(product, [3, 4], sys_dims);
        
        % Trace norm minimization
        dim_output = dYp * dXp;
        variable P(dim_output, dim_output) hermitian semidefinite
        variable N(dim_output, dim_output) hermitian semidefinite
        
        Theta_Phi - J_Psi == P - N;
        
        minimize(trace(P) + trace(N))
    cvx_end
    
    distance = cvx_optval;
    
    fprintf('  Status: %s\n', cvx_status);
    fprintf('  Distance: %.6f\n', distance);
    
    if distance < 1e-6
        fprintf('  ✓ EXACT CONVERSION POSSIBLE\n');
        % Create valid MATLAB variable name (only letters, numbers, underscores)
        varname = regexprep(description, '[^a-zA-Z0-9_]', '_');
        varname = ['J_Theta_' varname];
        assignin('base', varname, full(J_Theta));
        fprintf('  Saved to workspace as: %s\n', varname);
    elseif distance < 1.0
        fprintf('  ≈ APPROXIMATE CONVERSION (fidelity ~%.1f%%)\n', (1-distance/2)*100);
    else
        fprintf('  ✗ CONVERSION NOT POSSIBLE\n');
    end
end

function J_const = create_constant_channel(rho_target)
    % Create Choi matrix for constant channel: Φ(ρ) = rho_target for all ρ
    % This channel always outputs the same state regardless of input
    
    dim = size(rho_target, 1);
    
    % For constant channel, J = |ψ⟩⟨ψ| where |ψ⟩ = vec(√d * rho_target)
    % Actually: J(const) = I ⊗ rho_target (unnormalized)
    % Properly normalized: J = d * I ⊗ rho_target / Tr(I) = I ⊗ rho_target
    
    I = eye(dim);
    J_const = kron(I, rho_target);
    
    % Normalize properly for CPTP
    % For a channel Φ(ρ) = rho_target, the Choi is:
    % J = sum_i |i⟩⟨i| ⊗ rho_target = I ⊗ rho_target
    J_const = kron(I, rho_target);
end

function H = create_double_well_hamiltonian(a, b)
    % Create Hamiltonian H = (a/2)*σ_z + (b/2)*σ_x
    sigma_z = [1, 0; 0, -1];
    sigma_x = [0, 1; 1, 0];
    H = (a/2) * sigma_z + (b/2) * sigma_x;
end

function J_choi = create_channel(channel_type, dim, param)
    % Wrapper for channel creation (reuse from main file)
    switch channel_type
        case 'hamiltonian'
            H = param.H;
            t = param.t;
            U = expm(-1i * H * t);
            Kraus = {U};
            J_choi = kraus_to_choi(Kraus, dim);
        otherwise
            error('Unknown channel type');
    end
end

function J_choi = kraus_to_choi(Kraus, dim)
    % Convert Kraus operators to Choi matrix
    psi = MaxEntangled(dim, 1);
    J_choi = zeros(dim^2, dim^2);
    for i = 1:length(Kraus)
        K_i = Kraus{i};
        IK = kron(eye(dim), K_i);
        psi_i = IK * psi;
        J_choi = J_choi + psi_i * psi_i';
    end
end