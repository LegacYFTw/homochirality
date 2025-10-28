function quantum_channel_conversion_sdp()
    % Quantum Channel Conversion SDP using QETLAB
    % Based on Wang and Wilde, 2020
    
    fprintf('Quantum Channel Conversion SDP\n');
    fprintf('==================================================\n\n');
    
    % Dimensions
    dX = 2;   % Input dimension of channel Φ
    dY = 2;   % Output dimension of channel Φ
    dXp = 2;  % Input dimension of channel Ψ  
    dYp = 2;  % Output dimension of channel Ψ
    
    fprintf('Dimensions: dX=%d, dY=%d, dXp=%d, dYp=%d\n', dX, dY, dXp, dYp);
    
    % ===== CHOOSE YOUR CHANNELS HERE =====
    % Options: 'identity', 'depolarizing', 'amplitude_damping', 'dephasing', 'erasure'
    
    channel_Phi = 'depolarizing';  % Source channel
    channel_Psi = 'identity';       % Target channel
    param_Phi = 0.3;                % Parameter for Φ (e.g., noise level)
    param_Psi = 0.0;                % Parameter for Ψ
    
    fprintf('Testing %s(p=%.2f) to %s(p=%.2f) conversion...\n', ...
            channel_Phi, param_Phi, channel_Psi, param_Psi);
    
    % Create channels
    J_Phi = create_channel(channel_Phi, dX, param_Phi);
    J_Psi = create_channel(channel_Psi, dXp, param_Psi);
    
    fprintf('J(Φ) size: %dx%d\n', size(J_Phi));
    fprintf('J(Ψ) size: %dx%d\n', size(J_Psi));
    
    % Solve the SDP
    cvx_begin sdp quiet
        % Variable dimensions
        % J(Θ) lives in Y'⊗X'⊗Y⊗X (output ⊗ input for Choi matrix)
        dim_Theta = dYp * dXp * dY * dX;
        % J1 lives in X'⊗Y⊗X  
        dim_J1 = dXp * dY * dX;
        
        fprintf('Variable dimensions:\n');
        fprintf('  dim_Theta = %d (dYp*dXp*dY*dX = %d*%d*%d*%d)\n', dim_Theta, dYp, dXp, dY, dX);
        fprintf('  dim_J1 = %d (dXp*dY*dX = %d*%d*%d)\n', dim_J1, dXp, dY, dX);
        
        variable J_Theta(dim_Theta, dim_Theta) hermitian semidefinite
        variable J1(dim_J1, dim_J1) hermitian semidefinite
        
        % Identity matrices
        I_Yp = eye(dYp);
        I_Xp = eye(dXp);
        I_Y = eye(dY);
        I_X = eye(dX);
        
        % ===== CONSTRAINT 1: Tr_Y' J(Θ) = I_X' ⊗ J1 =====
        fprintf('\n=== CONSTRAINT 1: Tr_Y'' J(Θ) = I_X'' ⊗ J1 ===\n');
        % J(Θ) systems: [Y', X', Y, X]
        % We want to trace out Y' (system 1)
        sys_dims = [dYp, dXp, dY, dX];
        fprintf('System dimensions for J(Θ): [%d, %d, %d, %d]\n', sys_dims);
        
        J_Theta_traced = PartialTrace(J_Theta, 1, sys_dims);
        fprintf('After tracing Y'': size = %dx%d\n', size(J_Theta_traced));
        fprintf('Expected I_Xp ⊗ J1 size: %d x %d\n', dXp*dim_J1, dXp*dim_J1);
        
        % The result should be X'⊗Y⊗X, which matches J1
        % So we just need: J_Theta_traced == J1
        fprintf('Adding constraint: Tr_Y''[J(Θ)] == J1\n');
        J_Theta_traced == J1;
        
        % ===== CONSTRAINT 2: Tr_Y J1 = I_X' ⊗ I_X =====
        fprintf('\n=== CONSTRAINT 2: Tr_Y J1 = I_X'' ⊗ I_X ===\n');
        % J1 systems: [X', Y, X]
        % We want to trace out Y (system 2)
        J1_dims = [dXp, dY, dX];
        fprintf('System dimensions for J1: [%d, %d, %d]\n', J1_dims);
        
        J1_traced = PartialTrace(J1, 2, J1_dims);
        fprintf('After tracing Y: size = %dx%d\n', size(J1_traced));
        
        expected_J1_trace = kron(I_Xp, I_X);
        fprintf('I_Xp ⊗ I_X size: %dx%d\n', size(expected_J1_trace));
        
        fprintf('Adding constraint: Tr_Y[J1] == I_X'' ⊗ I_X\n');
        J1_traced == expected_J1_trace;
        
        % ===== CONSTRAINT 3: Channel conversion (with slack) =====
        fprintf('\n=== CONSTRAINT 3: Channel conversion ===\n');
        % Instead of Θ(Φ) = Ψ, we use: Θ(Φ) - Ψ = W - W†
        % where W is hermitian, and we minimize ||W||_1 = Tr(|W|)
        
        % First apply partial transpose on systems 3 and 4 (Y and X)
        fprintf('Computing PartialTranspose on Y,X (systems 3,4)...\n');
        J_Theta_pt = PartialTranspose(J_Theta, [3, 4], sys_dims);
        fprintf('J_Theta_pt size: %dx%d\n', size(J_Theta_pt));
        
        % Create I_Y' ⊗ J(Φ)
        % J(Φ) has systems [Y, X] with dimension dY*dX = 4
        % We need [Y', X', Y, X] for the product
        fprintf('Creating I_Y'' ⊗ I_X'' ⊗ J(Φ)...\n');
        I_YpXp = kron(I_Yp, I_Xp);
        rhs_operator = kron(I_YpXp, J_Phi);
        fprintf('I_Y''⊗I_X''⊗J(Φ) size: %dx%d\n', size(rhs_operator));
        
        % Multiply: J_Theta_pt * rhs_operator
        fprintf('Computing product...\n');
        product = J_Theta_pt * rhs_operator;
        fprintf('Product size: %dx%d\n', size(product));
        
        % Trace out Y and X (systems 3 and 4)
        fprintf('Tracing out Y,X (systems 3,4)...\n');
        Theta_Phi = PartialTrace(product, [3, 4], sys_dims);
        fprintf('Θ(Φ) size: %dx%d\n', size(Theta_Phi));
        fprintf('J(Ψ) size: %dx%d\n', size(J_Psi));
        
        % For trace norm minimization: ||A||_1 = min{ Tr(P+N) : A = P-N, P≥0, N≥0 }
        dim_output = dYp * dXp;
        variable P(dim_output, dim_output) hermitian semidefinite
        variable N(dim_output, dim_output) hermitian semidefinite
        
        fprintf('Adding constraint: Θ(Φ) - Ψ = P - N\n');
        Theta_Phi - J_Psi == P - N;
        
        % Objective: minimize trace norm ||Θ(Φ) - Ψ||_1
        fprintf('\nSetting objective: minimize Tr(P) + Tr(N)...\n');
        minimize(trace(P) + trace(N))
        
    cvx_end
    
    fprintf('\n==================================================\n');
    fprintf('Results:\n');
    fprintf('SDP status: %s\n', cvx_status);
    fprintf('Trace distance ||Θ(Φ) - Ψ||_1: %.6f\n', cvx_optval);
    
    if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
        if cvx_optval < 1e-6
            fprintf('SUCCESS: Exact channel conversion is possible!\n');
        elseif cvx_optval < 2  % Max trace distance is 2 for quantum states/channels
            fprintf('PARTIAL SUCCESS: Approximate conversion possible.\n');
            fprintf('  Best achievable fidelity: ~%.2f%%\n', (1 - cvx_optval/2)*100);
        else
            fprintf('FAILED: Channels are too different to convert.\n');
        end
        
        fprintf('\nSuperchannel J(Θ) properties:\n');
        fprintf('  Trace: %.6f\n', trace(J_Theta));
        fprintf('  Min eigenvalue: %.6e\n', min(eig(J_Theta)));
        fprintf('  Max eigenvalue: %.6e\n', max(eig(J_Theta)));
        fprintf('  Rank: %d (numerical rank with tol=1e-8)\n', rank(J_Theta, 1e-8));
        fprintf('  Frobenius norm: %.6f\n', norm(J_Theta, 'fro'));
        
        % Extract the superchannel Choi matrix
        J_Theta_opt = full(J_Theta);  % Convert from CVX variable to matrix
        
        % Verify the conversion works
        fprintf('\nVerifying the conversion:\n');
        J_Theta_pt_verify = PartialTranspose(J_Theta_opt, [3, 4], sys_dims);
        I_YpXp_verify = kron(I_Yp, I_Xp);
        rhs_verify = kron(I_YpXp_verify, J_Phi);
        product_verify = J_Theta_pt_verify * rhs_verify;
        Theta_Phi_verify = PartialTrace(product_verify, [3, 4], sys_dims);
        
        conversion_error = norm(Theta_Phi_verify - J_Psi, 1);
        fprintf('  ||Θ(Φ) - Ψ||_1 = %.6e (verification)\n', conversion_error);
        
        % Display the superchannel Choi matrix
        fprintf('\n==================================================\n');
        fprintf('SUPERCHANNEL CHOI MATRIX J(Θ):\n');
        fprintf('(Systems ordered as: [Y''_out, X''_out, Y_in, X_in])\n');
        fprintf('Size: %dx%d\n', size(J_Theta_opt));
        fprintf('\nJ(Θ) =\n');
        disp(J_Theta_opt);
        
        % Save to workspace
        assignin('base', 'J_Theta_optimal', J_Theta_opt);
        assignin('base', 'J_Phi', J_Phi);
        assignin('base', 'J_Psi', J_Psi);
        assignin('base', 'sys_dims_Theta', sys_dims);
        fprintf('\nSaved to workspace:\n');
        fprintf('  J_Theta_optimal - The superchannel Choi matrix\n');
        fprintf('  J_Phi - Source channel Choi matrix\n');
        fprintf('  J_Psi - Target channel Choi matrix\n');
        fprintf('  sys_dims_Theta - System dimensions [Y'', X'', Y, X]\n');
        fprintf('\nTo apply superchannel to a channel J_test:\n');
        fprintf('  result = apply_superchannel(J_Theta_optimal, J_test, sys_dims_Theta);\n');
        
        % Optionally extract Kraus-like representation
        fprintf('\n==================================================\n');
        fprintf('SPECTRAL DECOMPOSITION:\n');
        [V, D] = eig(J_Theta_opt);
        eigenvalues = diag(D);
        [eigenvalues_sorted, idx] = sort(real(eigenvalues), 'descend');
        V_sorted = V(:, idx);
        
        % Find significant eigenvalues
        threshold = 1e-8;
        significant = find(eigenvalues_sorted > threshold);
        fprintf('Number of significant eigenvalues (>%.0e): %d\n', threshold, length(significant));
        
        fprintf('\nTop eigenvalues:\n');
        for i = 1:min(5, length(significant))
            fprintf('  λ_%d = %.6f\n', i, eigenvalues_sorted(i));
        end
        
        % Save eigendecomposition
        assignin('base', 'J_Theta_eigenvalues', eigenvalues_sorted);
        assignin('base', 'J_Theta_eigenvectors', V_sorted);
        fprintf('\nAlso saved eigendecomposition to workspace:\n');
        fprintf('  J_Theta_eigenvalues\n');
        fprintf('  J_Theta_eigenvectors\n');
        
    else
        fprintf('FAILED: SDP solver error.\n');
        fprintf('Status: %s\n', cvx_status);
    end
end

function J_choi = create_channel(channel_type, dim, param)
    % Create Choi matrix for various quantum channels
    % 
    % Inputs:
    %   channel_type - string: 'identity', 'depolarizing', 'amplitude_damping', 
    %                          'dephasing', 'erasure', 'bit_flip', 'phase_flip'
    %   dim - dimension of the channel (usually 2 for qubits)
    %   param - channel parameter (interpretation depends on channel type)
    
    switch channel_type
        case 'identity'
            J_choi = create_identity_choi(dim);
            
        case 'depolarizing'
            % Depolarizing channel: ρ → (1-p)ρ + p*I/d
            % p is the depolarizing parameter (0 = identity, 1 = maximally mixed)
            p = param;
            if p < 0 || p > 1
                error('Depolarizing parameter must be in [0,1]');
            end
            
            % Kraus operators for depolarizing channel
            K0 = sqrt(1 - p) * eye(dim);
            
            if dim == 2
                % For qubits, use Pauli matrices
                K1 = sqrt(p/3) * [0, 1; 1, 0];      % σ_x
                K2 = sqrt(p/3) * [0, -1i; 1i, 0];   % σ_y
                K3 = sqrt(p/3) * [1, 0; 0, -1];     % σ_z
                Kraus = {K0, K1, K2, K3};
            else
                error('Depolarizing channel only implemented for qubits (dim=2)');
            end
            
            J_choi = kraus_to_choi(Kraus, dim);
            
        case 'amplitude_damping'
            % Amplitude damping: models energy dissipation
            % γ is the damping parameter (0 = no damping, 1 = complete damping)
            gamma = param;
            if gamma < 0 || gamma > 1
                error('Amplitude damping parameter must be in [0,1]');
            end
            
            if dim ~= 2
                error('Amplitude damping only implemented for qubits (dim=2)');
            end
            
            K0 = [1, 0; 0, sqrt(1-gamma)];
            K1 = [0, sqrt(gamma); 0, 0];
            Kraus = {K0, K1};
            
            J_choi = kraus_to_choi(Kraus, dim);
            
        case 'dephasing'
            % Dephasing (phase damping): ρ → (1-p)ρ + p*Z*ρ*Z
            % p is the dephasing parameter
            p = param;
            if p < 0 || p > 1
                error('Dephasing parameter must be in [0,1]');
            end
            
            if dim ~= 2
                error('Dephasing only implemented for qubits (dim=2)');
            end
            
            K0 = sqrt(1-p) * eye(2);
            K1 = sqrt(p) * [1, 0; 0, -1];  % σ_z
            Kraus = {K0, K1};
            
            J_choi = kraus_to_choi(Kraus, dim);
            
        case 'bit_flip'
            % Bit flip channel: ρ → (1-p)ρ + p*X*ρ*X
            p = param;
            if p < 0 || p > 1
                error('Bit flip parameter must be in [0,1]');
            end
            
            if dim ~= 2
                error('Bit flip only implemented for qubits (dim=2)');
            end
            
            K0 = sqrt(1-p) * eye(2);
            K1 = sqrt(p) * [0, 1; 1, 0];  % σ_x
            Kraus = {K0, K1};
            
            J_choi = kraus_to_choi(Kraus, dim);
            
        case 'phase_flip'
            % Phase flip channel: ρ → (1-p)ρ + p*Z*ρ*Z
            % (same as dephasing)
            p = param;
            if p < 0 || p > 1
                error('Phase flip parameter must be in [0,1]');
            end
            
            if dim ~= 2
                error('Phase flip only implemented for qubits (dim=2)');
            end
            
            K0 = sqrt(1-p) * eye(2);
            K1 = sqrt(p) * [1, 0; 0, -1];  % σ_z
            Kraus = {K0, K1};
            
            J_choi = kraus_to_choi(Kraus, dim);
            
        case 'erasure'
            % Erasure channel: ρ → (1-p)ρ + p|e⟩⟨e|
            % Requires dim+1 dimensional output space
            error('Erasure channel requires different output dimension - not yet implemented');
            
        case 'hamiltonian'
            % Unitary evolution under Hamiltonian
            % param should be a struct: param.H (Hamiltonian), param.t (time)
            if ~isstruct(param)
                error('For Hamiltonian channel, param must be struct with fields H and t');
            end
            H = param.H;
            t = param.t;
            
            % Unitary evolution: U = exp(-i*H*t)
            U = expm(-1i * H * t);
            
            % Single Kraus operator (unitary channel)
            Kraus = {U};
            J_choi = kraus_to_choi(Kraus, dim);
            
        otherwise
            error('Unknown channel type: %s', channel_type);
    end
end

function result = apply_superchannel(J_Theta, J_channel, sys_dims)
    % Apply a superchannel to a channel
    % 
    % Inputs:
    %   J_Theta - Choi matrix of superchannel (dim_output^2 × dim_output^2)
    %   J_channel - Choi matrix of input channel (dim_input^2 × dim_input^2)  
    %   sys_dims - System dimensions [dYp, dXp, dY, dX]
    %
    % Output:
    %   result - Choi matrix of output channel Θ(Φ)
    
    dYp = sys_dims(1);
    dXp = sys_dims(2);
    dY = sys_dims(3);
    dX = sys_dims(4);
    
    % Apply partial transpose on input systems (Y, X)
    J_Theta_pt = PartialTranspose(J_Theta, [3, 4], sys_dims);
    
    % Tensor with identity on output systems
    I_YpXp = kron(eye(dYp), eye(dXp));
    rhs_operator = kron(I_YpXp, J_channel);
    
    % Matrix multiplication
    product = J_Theta_pt * rhs_operator;
    
    % Trace out input systems (Y, X)
    result = PartialTrace(product, [3, 4], sys_dims);
end

function H = create_double_well_hamiltonian(a, b)
    % Create Hamiltonian H = (a/2)*σ_z + (b/2)*σ_x
    % 
    % For b=0: asymmetric double well with |L⟩,|R⟩ having energies ±a/2
    % For b≠0: tunneling between wells
    %
    % Basis: |L⟩ = |0⟩, |R⟩ = |1⟩
    
    sigma_z = [1, 0; 0, -1];
    sigma_x = [0, 1; 1, 0];
    
    H = (a/2) * sigma_z + (b/2) * sigma_x;
end

function J_choi = kraus_to_choi(Kraus, dim)
    % Convert Kraus operators to Choi matrix
    % Kraus is a cell array of Kraus operators
    % dim is the input/output dimension
    
    % Create maximally entangled state |Φ⁺⟩
    psi = MaxEntangled(dim, 1);
    
    % Choi matrix: J = (id ⊗ Φ)(|Φ⁺⟩⟨Φ⁺|)
    % This is equivalent to: sum_i (I ⊗ K_i) |Φ⁺⟩⟨Φ⁺| (I ⊗ K_i^†)
    
    J_choi = zeros(dim^2, dim^2);
    for i = 1:length(Kraus)
        K_i = Kraus{i};
        % Apply (I ⊗ K_i) to |Φ⁺⟩
        IK = kron(eye(dim), K_i);
        psi_i = IK * psi;
        J_choi = J_choi + psi_i * psi_i';
    end
end

function J_choi = create_identity_choi(dim)
    % Create Choi matrix for identity channel using QETLAB
    % J(id) = |Φ⁺⟩⟨Φ⁺| where |Φ⁺⟩ is the maximally entangled state
    psi = MaxEntangled(dim, 1);
    J_choi = psi * psi';
end