function parameter_sweep_analysis()
    % Comprehensive parameter sweep for homochirality conversion
    % Analyzes optimal superchannels and their physical characteristics
    
    close all;
    
    fprintf('=========================================\n');
    fprintf('PARAMETER SWEEP: HOMOCHIRALITY CONVERSION\n');
    fprintf('=========================================\n\n');
    
    % Parameter ranges
    a_values = linspace(0.1, 2.0, 15);  % Energy asymmetry
    b_values = linspace(0.0, 1.0, 15);  % Tunneling amplitude
    t_values = linspace(0.1, 5.0, 15);  % Evolution time
    
    % Storage for results
    results = struct();
    
    %% SWEEP 1: Energy asymmetry (a) vs Time (t), fixed b=0
    fprintf('\n=== SWEEP 1: Energy Asymmetry vs Time (b=0) ===\n');
    b_fixed = 0.0;
    
    distance_L_at = zeros(length(a_values), length(t_values));
    distance_R_at = zeros(length(a_values), length(t_values));
    best_channels_at = cell(length(a_values), length(t_values));
    
    for i = 1:length(a_values)
        for j = 1:length(t_values)
            a = a_values(i);
            t = t_values(j);
            
            H = create_double_well_hamiltonian(a, b_fixed);
            param_H.H = H;
            param_H.t = t;
            J_Phi = create_channel('hamiltonian', 2, param_H);
            
            % Test conversion to |L⟩
            rho_L = [1, 0; 0, 0];
            J_Psi_L = create_constant_channel(rho_L);
            [dist_L, J_Theta_L] = solve_conversion_sdp_silent(J_Phi, J_Psi_L);
            distance_L_at(i,j) = dist_L;
            
            % Test conversion to |R⟩
            rho_R = [0, 0; 0, 1];
            J_Psi_R = create_constant_channel(rho_R);
            [dist_R, J_Theta_R] = solve_conversion_sdp_silent(J_Phi, J_Psi_R);
            distance_R_at(i,j) = dist_R;
            
            % Store best channel
            if dist_L < dist_R && dist_L < 0.1
                best_channels_at{i,j} = struct('Theta', J_Theta_L, 'target', 'L', 'distance', dist_L);
            elseif dist_R < 0.1
                best_channels_at{i,j} = struct('Theta', J_Theta_R, 'target', 'R', 'distance', dist_R);
            end
        end
        fprintf('Progress: a = %.2f/%2.f\n', a, a_values(end));
    end
    
    %% SWEEP 2: Tunneling (b) vs Time (t), fixed a=1.0
    fprintf('\n=== SWEEP 2: Tunneling vs Time (a=1.0) ===\n');
    a_fixed = 1.0;
    
    distance_L_bt = zeros(length(b_values), length(t_values));
    distance_R_bt = zeros(length(b_values), length(t_values));
    best_channels_bt = cell(length(b_values), length(t_values));
    
    for i = 1:length(b_values)
        for j = 1:length(t_values)
            b = b_values(i);
            t = t_values(j);
            
            H = create_double_well_hamiltonian(a_fixed, b);
            param_H.H = H;
            param_H.t = t;
            J_Phi = create_channel('hamiltonian', 2, param_H);
            
            rho_L = [1, 0; 0, 0];
            J_Psi_L = create_constant_channel(rho_L);
            [dist_L, J_Theta_L] = solve_conversion_sdp_silent(J_Phi, J_Psi_L);
            distance_L_bt(i,j) = dist_L;
            
            rho_R = [0, 0; 0, 1];
            J_Psi_R = create_constant_channel(rho_R);
            [dist_R, J_Theta_R] = solve_conversion_sdp_silent(J_Phi, J_Psi_R);
            distance_R_bt(i,j) = dist_R;
            
            if dist_L < dist_R && dist_L < 0.1
                best_channels_bt{i,j} = struct('Theta', J_Theta_L, 'target', 'L', 'distance', dist_L);
            elseif dist_R < 0.1
                best_channels_bt{i,j} = struct('Theta', J_Theta_R, 'target', 'R', 'distance', dist_R);
            end
        end
        fprintf('Progress: b = %.2f/%.2f\n', b, b_values(end));
    end
    
    %% ANALYZE OPTIMAL CHANNELS
    fprintf('\n=== ANALYZING OPTIMAL SUPERCHANNEL PROPERTIES ===\n');
    
    % Find best conversion points
    [min_dist_L_at, idx_L_at] = min(distance_L_at(:));
    [i_opt_at, j_opt_at] = ind2sub(size(distance_L_at), idx_L_at);
    a_opt = a_values(i_opt_at);
    t_opt = t_values(j_opt_at);
    
    fprintf('Optimal parameters (a vs t sweep):\n');
    fprintf('  a = %.3f, t = %.3f\n', a_opt, t_opt);
    fprintf('  Distance to |L⟩: %.6f\n', min_dist_L_at);
    
    % Get optimal superchannel
    if ~isempty(best_channels_at{i_opt_at, j_opt_at})
        J_Theta_opt = best_channels_at{i_opt_at, j_opt_at}.Theta;
        
        % Analyze this superchannel
        channel_props = analyze_superchannel(J_Theta_opt);
        
        % Test on different input states
        test_states = {
            [1, 0; 0, 0], '|L⟩';
            [0, 0; 0, 1], '|R⟩';
            0.5*[1, 1; 1, 1], '|+⟩';
            0.5*[1, -1i; 1i, 1], '|+i⟩';
            0.5*eye(2), 'I/2'
        };
        
        fprintf('\nTesting optimal superchannel on various input states:\n');
        H_opt = create_double_well_hamiltonian(a_opt, b_fixed);
        param_H_opt.H = H_opt;
        param_H_opt.t = t_opt;
        
        output_states = cell(length(test_states), 1);
        for k = 1:length(test_states)
            rho_in = test_states{k, 1};
            state_name = test_states{k, 2};
            
            rho_out = apply_channel_to_state(J_Theta_opt, H_opt, t_opt, rho_in);
            output_states{k} = rho_out;
            
            fprintf('  %s → ρ_out, Tr(ρ)=%.4f, Purity=%.4f\n', ...
                state_name, trace(rho_out), trace(rho_out^2));
            fprintf('    Populations: [%.4f, %.4f]\n', rho_out(1,1), rho_out(2,2));
        end
    end
    
    %% PLOTTING
    fprintf('\n=== GENERATING PLOTS ===\n');
    
    % Figure 1: Distance heatmaps
    figure('Position', [100, 100, 1400, 500]);
    
    subplot(1,3,1);
    imagesc(t_values, a_values, distance_L_at);
    colorbar;
    xlabel('Time t');
    ylabel('Energy asymmetry a');
    title('Distance to |L⟩⟨L| (b=0)');
    set(gca, 'YDir', 'normal');
    colormap(gca, 'hot');
    caxis([0, 2]);
    
    subplot(1,3,2);
    imagesc(t_values, a_values, distance_R_at);
    colorbar;
    xlabel('Time t');
    ylabel('Energy asymmetry a');
    title('Distance to |R⟩⟨R| (b=0)');
    set(gca, 'YDir', 'normal');
    colormap(gca, 'hot');
    caxis([0, 2]);
    
    subplot(1,3,3);
    min_distance_at = min(distance_L_at, distance_R_at);
    imagesc(t_values, a_values, min_distance_at);
    colorbar;
    xlabel('Time t');
    ylabel('Energy asymmetry a');
    title('Best achievable distance (b=0)');
    set(gca, 'YDir', 'normal');
    colormap(gca, 'hot');
    caxis([0, 2]);
    
    % Mark optimal point
    hold on;
    plot(t_opt, a_opt, 'c*', 'MarkerSize', 15, 'LineWidth', 2);
    hold off;
    
    sgtitle('Conversion Distance: Hamiltonian Evolution → Homochiral State');
    
    % Figure 2: Tunneling sweep
    figure('Position', [100, 100, 1400, 500]);
    
    subplot(1,3,1);
    imagesc(t_values, b_values, distance_L_bt);
    colorbar;
    xlabel('Time t');
    ylabel('Tunneling b');
    title('Distance to |L⟩⟨L| (a=1)');
    set(gca, 'YDir', 'normal');
    colormap(gca, 'hot');
    caxis([0, 2]);
    
    subplot(1,3,2);
    imagesc(t_values, b_values, distance_R_bt);
    colorbar;
    xlabel('Time t');
    ylabel('Tunneling b');
    title('Distance to |R⟩⟨R| (a=1)');
    set(gca, 'YDir', 'normal');
    colormap(gca, 'hot');
    caxis([0, 2]);
    
    subplot(1,3,3);
    min_distance_bt = min(distance_L_bt, distance_R_bt);
    imagesc(t_values, b_values, min_distance_bt);
    colorbar;
    xlabel('Time t');
    ylabel('Tunneling b');
    title('Best achievable distance (a=1)');
    set(gca, 'YDir', 'normal');
    colormap(gca, 'hot');
    caxis([0, 2]);
    
    sgtitle('Conversion Distance with Tunneling: H(a=1,b,t) → Homochiral State');
    
    % Figure 3: Superchannel properties
    if ~isempty(best_channels_at{i_opt_at, j_opt_at})
        figure('Position', [100, 100, 1200, 800]);
        
        subplot(2,3,1);
        bar([channel_props.dephasing_rate, channel_props.dissipation_rate]);
        set(gca, 'XTickLabel', {'Dephasing', 'Dissipation'});
        ylabel('Rate');
        title('Decoherence Rates');
        grid on;
        
        subplot(2,3,2);
        eigenvals = channel_props.eigenvalues;
        bar(real(eigenvals(eigenvals > 1e-10)));
        xlabel('Eigenvalue index');
        ylabel('Eigenvalue');
        title('Superchannel Spectrum');
        grid on;
        
        subplot(2,3,3);
        pie([channel_props.unitary_weight, channel_props.nonunitary_weight], ...
            {'Unitary', 'Non-unitary'});
        title('Unitary vs Non-unitary Weight');
        
        subplot(2,3,4);
        test_names = {'|L⟩', '|R⟩', '|+⟩', '|+i⟩', 'I/2'};
        populations_L = zeros(length(output_states), 1);
        populations_R = zeros(length(output_states), 1);
        for k = 1:length(output_states)
            populations_L(k) = real(output_states{k}(1,1));
            populations_R(k) = real(output_states{k}(2,2));
        end
        bar([populations_L, populations_R]);
        set(gca, 'XTickLabel', test_names);
        ylabel('Population');
        legend('|L⟩', '|R⟩');
        title('Output Populations');
        grid on;
        
        subplot(2,3,5);
        coherences = zeros(length(output_states), 1);
        for k = 1:length(output_states)
            coherences(k) = abs(output_states{k}(1,2));
        end
        bar(coherences);
        set(gca, 'XTickLabel', test_names);
        ylabel('|ρ_{LR}|');
        title('Output Coherences');
        grid on;
        
        subplot(2,3,6);
        purities = zeros(length(output_states), 1);
        for k = 1:length(output_states)
            purities(k) = real(trace(output_states{k}^2));
        end
        bar(purities);
        set(gca, 'XTickLabel', test_names);
        ylabel('Tr(ρ²)');
        ylim([0, 1]);
        title('Output State Purity');
        grid on;
        
        sgtitle(sprintf('Optimal Superchannel Analysis (a=%.2f, t=%.2f)', a_opt, t_opt));
    end
    
    % Figure 4: Bloch sphere visualization
    if ~isempty(best_channels_at{i_opt_at, j_opt_at})
        figure('Position', [100, 100, 800, 800]);
        
        % Plot Bloch sphere
        [x, y, z] = sphere(30);
        surf(x, y, z, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
        hold on;
        
        % Plot axes
        plot3([0 1.2], [0 0], [0 0], 'r-', 'LineWidth', 2);
        plot3([0 0], [0 1.2], [0 0], 'g-', 'LineWidth', 2);
        plot3([0 0], [0 0], [0 1.2], 'b-', 'LineWidth', 2);
        text(1.3, 0, 0, 'X', 'FontSize', 14);
        text(0, 1.3, 0, 'Y', 'FontSize', 14);
        text(0, 0, 1.3, 'Z', 'FontSize', 14);
        
        % Plot input and output states
        colors = {'r', 'b', 'g', 'm', 'c'};
        markers = {'o', 's', 'd', '^', 'v'};
        
        for k = 1:length(test_states)
            rho_in = test_states{k, 1};
            rho_out = output_states{k};
            
            % Convert to Bloch vectors
            [x_in, y_in, z_in] = density_to_bloch(rho_in);
            [x_out, y_out, z_out] = density_to_bloch(rho_out);
            
            % Plot input state
            plot3(x_in, y_in, z_in, [colors{k} markers{k}], ...
                'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', colors{k});
            
            % Plot output state
            plot3(x_out, y_out, z_out, [colors{k} markers{k}], ...
                'MarkerSize', 15, 'LineWidth', 2);
            
            % Draw arrow
            quiver3(x_in, y_in, z_in, x_out-x_in, y_out-y_in, z_out-z_in, ...
                0, 'Color', colors{k}, 'LineWidth', 1.5);
        end
        
        % Mark target state
        plot3(0, 0, 1, 'k*', 'MarkerSize', 20, 'LineWidth', 3);
        text(0, 0, 1.15, 'Target |L⟩', 'FontSize', 12, 'HorizontalAlignment', 'center');
        
        axis equal;
        xlabel('X'); ylabel('Y'); zlabel('Z');
        title('Bloch Sphere: State Evolution under Optimal Superchannel');
        legend(test_names, 'Location', 'best');
        grid on;
        view(45, 30);
        hold off;
    end
    
    % Save results
    results.a_values = a_values;
    results.b_values = b_values;
    results.t_values = t_values;
    results.distance_L_at = distance_L_at;
    results.distance_R_at = distance_R_at;
    results.distance_L_bt = distance_L_bt;
    results.distance_R_bt = distance_R_bt;
    results.optimal_params = struct('a', a_opt, 't', t_opt, 'distance', min_dist_L_at);
    if ~isempty(best_channels_at{i_opt_at, j_opt_at})
        results.optimal_superchannel = J_Theta_opt;
        results.channel_properties = channel_props;
    end
    
    assignin('base', 'sweep_results', results);
    fprintf('\nResults saved to workspace as ''sweep_results''\n');
end

function [distance, J_Theta] = solve_conversion_sdp_silent(J_Phi, J_Psi)
    % Solve SDP without output
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
        
        sys_dims = [dYp, dXp, dY, dX];
        J_Theta_traced = PartialTrace(J_Theta, 1, sys_dims);
        J_Theta_traced == J1;
        
        J1_dims = [dXp, dY, dX];
        J1_traced = PartialTrace(J1, 2, J1_dims);
        J1_traced == kron(I_Xp, I_X);
        
        J_Theta_pt = PartialTranspose(J_Theta, [3, 4], sys_dims);
        I_YpXp = kron(I_Yp, I_Xp);
        rhs_operator = kron(I_YpXp, J_Phi);
        product = J_Theta_pt * rhs_operator;
        Theta_Phi = PartialTrace(product, [3, 4], sys_dims);
        
        dim_output = dYp * dXp;
        variable P(dim_output, dim_output) hermitian semidefinite
        variable N(dim_output, dim_output) hermitian semidefinite
        
        Theta_Phi - J_Psi == P - N;
        
        minimize(trace(P) + trace(N))
    cvx_end
    
    distance = cvx_optval;
    J_Theta = full(J_Theta);
end

function props = analyze_superchannel(J_Theta)
    % Analyze physical properties of the superchannel
    
    % Eigenvalue decomposition
    [V, D] = eig(J_Theta);
    eigenvalues = sort(real(diag(D)), 'descend');
    
    % Trace and purity
    tr_Theta = trace(J_Theta);
    purity = trace(J_Theta^2) / tr_Theta^2;
    
    % Estimate dephasing and dissipation rates
    % These are heuristic measures based on Choi matrix structure
    dim = size(J_Theta, 1);
    n = sqrt(dim);  % Should be 4 for qubit channels
    
    % Decompose into blocks
    J_reshaped = reshape(J_Theta, [n, n, n, n]);
    
    % Dephasing: loss of off-diagonal coherences
    diagonal_weight = 0;
    offdiag_weight = 0;
    for i = 1:n
        for j = 1:n
            block = squeeze(J_reshaped(i, j, :, :));
            if i == j
                diagonal_weight = diagonal_weight + norm(block, 'fro')^2;
            else
                offdiag_weight = offdiag_weight + norm(block, 'fro')^2;
            end
        end
    end
    
    total_weight = diagonal_weight + offdiag_weight;
    dephasing_rate = 1 - offdiag_weight / total_weight;
    
    % Dissipation: population transfer
    % Measure asymmetry in population blocks
    pop_00 = norm(squeeze(J_reshaped(1, 1, :, :)), 'fro');
    pop_11 = norm(squeeze(J_reshaped(2, 2, :, :)), 'fro');
    dissipation_rate = abs(pop_00 - pop_11) / (pop_00 + pop_11);
    
    % Unitary weight: largest eigenvalue / trace
    unitary_weight = eigenvalues(1) / tr_Theta;
    nonunitary_weight = 1 - unitary_weight;
    
    props = struct(...
        'eigenvalues', eigenvalues, ...
        'trace', tr_Theta, ...
        'purity', purity, ...
        'dephasing_rate', dephasing_rate, ...
        'dissipation_rate', dissipation_rate, ...
        'unitary_weight', unitary_weight, ...
        'nonunitary_weight', nonunitary_weight ...
    );
end

function rho_out = apply_channel_to_state(J_Theta, H, t, rho_in)
    % Apply superchannel(Hamiltonian evolution) to a state
    
    % First apply Hamiltonian evolution
    U = expm(-1i * H * t);
    rho_mid = U * rho_in * U';
    
    % Then apply the superchannel's effective action
    % This is approximate - proper application would need the full superchannel machinery
    % For now, extract an effective channel from J_Theta
    
    % Create Choi for evolved channel
    param.H = H;
    param.t = t;
    J_Phi = create_channel('hamiltonian', 2, param);
    
    % Apply superchannel to get output channel
    sys_dims = [2, 2, 2, 2];
    J_Theta_pt = PartialTranspose(J_Theta, [3, 4], sys_dims);
    I_YpXp = kron(eye(2), eye(2));
    rhs_operator = kron(I_YpXp, J_Phi);
    product = J_Theta_pt * rhs_operator;
    J_output = PartialTrace(product, [3, 4], sys_dims);
    
    % Apply output channel to input state
    rho_out = apply_choi_to_state(J_output, rho_in);
end

function rho_out = apply_choi_to_state(J_choi, rho_in)
    % Apply a channel (given by Choi matrix) to a state
    % Φ(ρ) = Tr_in[(I ⊗ ρ^T) J]
    
    dim = size(rho_in, 1);
    rho_in_T = rho_in.';
    
    % Vectorize: |ρ⟩ = vec(ρ)
    rho_vec = reshape(rho_in_T, [dim^2, 1]);
    
    % Apply: |Φ(ρ)⟩ = J|ρ⟩
    out_vec = J_choi * rho_vec;
    
    % Unvectorize
    rho_out = reshape(out_vec, [dim, dim]).';
end

function [x, y, z] = density_to_bloch(rho)
    % Convert density matrix to Bloch vector
    sigma_x = [0, 1; 1, 0];
    sigma_y = [0, -1i; 1i, 0];
    sigma_z = [1, 0; 0, -1];
    
    x = real(trace(rho * sigma_x));
    y = real(trace(rho * sigma_y));
    z = real(trace(rho * sigma_z));
end

function H = create_double_well_hamiltonian(a, b)
    sigma_z = [1, 0; 0, -1];
    sigma_x = [0, 1; 1, 0];
    H = (a/2) * sigma_z + (b/2) * sigma_x;
end

function J_const = create_constant_channel(rho_target)
    dim = size(rho_target, 1);
    I = eye(dim);
    J_const = kron(I, rho_target);
end

function J_choi = create_channel(channel_type, dim, param)
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
    psi = MaxEntangled(dim, 1);
    J_choi = zeros(dim^2, dim^2);
    for i = 1:length(Kraus)
        K_i = Kraus{i};
        IK = kron(eye(dim), K_i);
        psi_i = IK * psi;
        J_choi = J_choi + psi_i * psi_i';
    end
end