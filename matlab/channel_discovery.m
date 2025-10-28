function sdp_discovery_comprehensive_analysis()
    % Comprehensive SDP-based channel discovery for homochirality
    % Discovers optimal channels via SDP, then analyzes their properties
    
    close all;
    
    fprintf('=========================================\n');
    fprintf('SDP-BASED CHANNEL DISCOVERY FOR HOMOCHIRALITY\n');
    fprintf('=========================================\n\n');
    
    %% PARAMETER RANGES
    a_values = linspace(0.1, 2.0, 12);      % Energy asymmetry
    b_values = linspace(0.0, 1.0, 12);      % Tunneling amplitude
    t_values = linspace(0.5, 5.0, 12);      % Evolution time
    
    % Target states
    targets = {
        'L', [1, 0; 0, 0];      % Left homochiral
        'R', [0, 0; 0, 1];      % Right homochiral
        'plus', 0.5*[1, 1; 1, 1];  % Superposition
    };
    
    %% COMPREHENSIVE PARAMETER SWEEP
    fprintf('Running comprehensive parameter sweep...\n');
    fprintf('Total combinations: %d\n', length(a_values)*length(b_values)*length(t_values)*size(targets,1));
    
    results = [];
    discovered_channels = {};
    
    counter = 0;
    total = length(a_values)*length(b_values)*length(t_values)*size(targets,1);
    
    for i_a = 1:length(a_values)
        for i_b = 1:length(b_values)
            for i_t = 1:length(t_values)
                a = a_values(i_a);
                b = b_values(i_b);
                t = t_values(i_t);
                
                % Create Hamiltonian and its channel
                H = create_double_well_hamiltonian(a, b);
                param_H.H = H;
                param_H.t = t;
                J_Phi = create_channel('hamiltonian', 2, param_H);
                
                for i_target = 1:size(targets, 1)
                    target_name = targets{i_target, 1};
                    rho_target = targets{i_target, 2};
                    J_Psi = create_constant_channel(rho_target);
                    
                    % SOLVE SDP TO DISCOVER OPTIMAL CHANNEL
                    [distance, J_Theta, status] = solve_conversion_sdp_silent(J_Phi, J_Psi);
                    
                    counter = counter + 1;
                    if mod(counter, 50) == 0
                        fprintf('Progress: %d/%d (%.1f%%)\n', counter, total, 100*counter/total);
                    end
                    
                    % Store result
                    result = struct();
                    result.a = a;
                    result.b = b;
                    result.t = t;
                    result.target = target_name;
                    result.distance = distance;
                    result.success = (distance < 0.05);  % Threshold for success
                    result.status = status;
                    
                    if result.success
                        % ANALYZE THE DISCOVERED CHANNEL
                        channel_analysis = analyze_discovered_channel(J_Theta, J_Phi);
                        
                        result.dephasing = channel_analysis.dephasing_rate;
                        result.dissipation = channel_analysis.dissipation_rate;
                        result.coherence_loss = channel_analysis.coherence_loss;
                        result.pop_asymmetry = channel_analysis.population_asymmetry;
                        result.unitary_weight = channel_analysis.unitary_weight;
                        result.kraus_rank = channel_analysis.kraus_rank;
                        
                        % Store the discovered channel
                        channel_key = sprintf('a%.2f_b%.2f_t%.2f_%s', a, b, t, target_name);
                        discovered_channels.(matlab.lang.makeValidName(channel_key)) = J_Theta;
                    else
                        % Fill with NaN for failed conversions
                        result.dephasing = NaN;
                        result.dissipation = NaN;
                        result.coherence_loss = NaN;
                        result.pop_asymmetry = NaN;
                        result.unitary_weight = NaN;
                        result.kraus_rank = NaN;
                    end
                    
                    results = [results; result];
                end
            end
        end
    end
    
    fprintf('Sweep complete! Found %d successful conversions.\n', sum([results.success]));
    
    %% ANALYZE DISCOVERED CHANNELS
    fprintf('\nAnalyzing discovered channel properties...\n');
    
    % Extract successful cases
    success_idx = [results.success];
    successful_results = results(success_idx);
    
    if isempty(successful_results)
        fprintf('WARNING: No successful conversions found!\n');
        return;
    end
    
    %% DETAILED ANALYSIS OF BEST CHANNELS
    fprintf('\nFinding optimal channels for each target...\n');
    
    best_channels = struct();
    for i_target = 1:size(targets, 1)
        target_name = targets{i_target, 1};
        target_results = results(strcmp({results.target}, target_name) & success_idx);
        
        if ~isempty(target_results)
            [~, best_idx] = min([target_results.distance]);
            best_result = target_results(best_idx);
            
            fprintf('\nBest channel for target |%s⟩:\n', target_name);
            fprintf('  Parameters: a=%.2f, b=%.2f, t=%.2f\n', ...
                best_result.a, best_result.b, best_result.t);
            fprintf('  Distance: %.6f\n', best_result.distance);
            fprintf('  Dephasing rate: %.4f\n', best_result.dephasing);
            fprintf('  Dissipation rate: %.4f\n', best_result.dissipation);
            fprintf('  Coherence loss: %.4f\n', best_result.coherence_loss);
            fprintf('  Population asymmetry: %.4f\n', best_result.pop_asymmetry);
            fprintf('  Unitary weight: %.4f\n', best_result.unitary_weight);
            fprintf('  Kraus rank: %d\n', best_result.kraus_rank);
            
            best_channels.(target_name) = best_result;
        end
    end
    
    %% GENERATE COMPREHENSIVE VISUALIZATIONS
    fprintf('\nGenerating comprehensive visualizations...\n');
    
    % Convert to table for easier manipulation
    results_table = struct2table(results);
    
    % FIGURE 1: Parameter Space Heatmaps
    create_parameter_heatmaps(results_table, a_values, b_values, t_values);
    
    % FIGURE 2: Channel Properties Analysis
    create_channel_properties_plots(results_table);
    
    % FIGURE 3: Mechanism Classification
    create_mechanism_classification(results_table);
    
    % FIGURE 4: Optimal Channels Comparison
    if ~isempty(fieldnames(best_channels))
        create_optimal_channels_comparison(best_channels, discovered_channels);
    end
    
    % FIGURE 5: Success Rate Analysis
    create_success_rate_analysis(results_table);
    
    % FIGURE 6: Phase Diagrams
    create_phase_diagrams(results_table, a_values, b_values, t_values);
    
    %% SAVE RESULTS
    save('sdp_discovery_results.mat', 'results', 'discovered_channels', 'best_channels');
    assignin('base', 'sdp_results', results);
    assignin('base', 'discovered_channels', discovered_channels);
    assignin('base', 'best_channels', best_channels);
    
    fprintf('\n✅ ANALYSIS COMPLETE!\n');
    fprintf('Results saved to workspace and sdp_discovery_results.mat\n');
    
    %% PRINT SUMMARY STATISTICS
    print_summary_statistics(results_table, best_channels);
end

function create_parameter_heatmaps(results_table, a_values, b_values, t_values)
    % Create comprehensive heatmaps of parameter space
    
    figure('Position', [50, 50, 1600, 1000]);
    sgtitle('SDP Discovery: Parameter Space Exploration', 'FontSize', 16, 'FontWeight', 'bold');
    
    targets = unique(results_table.target);
    
    % For each target, create heatmaps
    for i_target = 1:length(targets)
        target = targets{i_target};
        target_data = results_table(strcmp(results_table.target, target), :);
        
        % a vs b (averaged over t)
        subplot(3, length(targets), i_target);
        success_map = zeros(length(a_values), length(b_values));
        for i = 1:length(a_values)
            for j = 1:length(b_values)
                mask = (abs(target_data.a - a_values(i)) < 1e-6) & ...
                       (abs(target_data.b - b_values(j)) < 1e-6);
                if sum(mask) > 0
                    success_map(i, j) = mean(target_data.success(mask));
                end
            end
        end
        imagesc(b_values, a_values, success_map);
        colorbar;
        xlabel('Tunneling b');
        ylabel('Energy asymmetry a');
        title(sprintf('Success Rate: Target |%s⟩', target));
        set(gca, 'YDir', 'normal');
        colormap(gca, flipud(hot));
        
        % a vs t (averaged over b)
        subplot(3, length(targets), length(targets) + i_target);
        distance_map = zeros(length(a_values), length(t_values));
        for i = 1:length(a_values)
            for j = 1:length(t_values)
                mask = (abs(target_data.a - a_values(i)) < 1e-6) & ...
                       (abs(target_data.t - t_values(j)) < 1e-6);
                if sum(mask) > 0
                    distances = target_data.distance(mask);
                    distance_map(i, j) = mean(distances(~isnan(distances)));
                else
                    distance_map(i, j) = 2; % Max distance
                end
            end
        end
        imagesc(t_values, a_values, distance_map);
        colorbar;
        xlabel('Time t');
        ylabel('Energy asymmetry a');
        title(sprintf('Avg Distance: Target |%s⟩', target));
        set(gca, 'YDir', 'normal');
        colormap(gca, hot);
        caxis([0, 2]);
        
        % b vs t (averaged over a)
        subplot(3, length(targets), 2*length(targets) + i_target);
        success_map_bt = zeros(length(b_values), length(t_values));
        for i = 1:length(b_values)
            for j = 1:length(t_values)
                mask = (abs(target_data.b - b_values(i)) < 1e-6) & ...
                       (abs(target_data.t - t_values(j)) < 1e-6);
                if sum(mask) > 0
                    success_map_bt(i, j) = mean(target_data.success(mask));
                end
            end
        end
        imagesc(t_values, b_values, success_map_bt);
        colorbar;
        xlabel('Time t');
        ylabel('Tunneling b');
        title(sprintf('Success Rate: Target |%s⟩', target));
        set(gca, 'YDir', 'normal');
        colormap(gca, flipud(hot));
    end
end

function create_channel_properties_plots(results_table)
    % Analyze properties of discovered channels
    
    figure('Position', [100, 100, 1600, 1000]);
    sgtitle('Discovered Channel Properties', 'FontSize', 16, 'FontWeight', 'bold');
    
    success_data = results_table(results_table.success == 1, :);
    
    if isempty(success_data)
        return;
    end
    
    % 1. Dephasing vs Dissipation scatter
    subplot(3, 4, 1);
    targets = unique(success_data.target);
    colors = lines(length(targets));
    for i = 1:length(targets)
        target_data = success_data(strcmp(success_data.target, targets{i}), :);
        scatter(target_data.dephasing, target_data.dissipation, 50, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.6);
        hold on;
    end
    xlabel('Dephasing Rate');
    ylabel('Dissipation Rate');
    title('Dephasing vs Dissipation');
    legend(targets, 'Location', 'best');
    grid on;
    
    % 2. Coherence loss distribution
    subplot(3, 4, 2);
    for i = 1:length(targets)
        target_data = success_data(strcmp(success_data.target, targets{i}), :);
        histogram(target_data.coherence_loss, 20, 'FaceAlpha', 0.5, 'DisplayName', targets{i});
        hold on;
    end
    xlabel('Coherence Loss');
    ylabel('Count');
    title('Coherence Loss Distribution');
    legend('Location', 'best');
    grid on;
    
    % 3. Unitary weight distribution
    subplot(3, 4, 3);
    for i = 1:length(targets)
        target_data = success_data(strcmp(success_data.target, targets{i}), :);
        histogram(target_data.unitary_weight, 20, 'FaceAlpha', 0.5, 'DisplayName', targets{i});
        hold on;
    end
    xlabel('Unitary Weight');
    ylabel('Count');
    title('Unitary vs Non-unitary');
    legend('Location', 'best');
    grid on;
    
    % 4. Kraus rank distribution
    subplot(3, 4, 4);
    kraus_ranks = unique(success_data.kraus_rank);
    for i = 1:length(targets)
        target_data = success_data(strcmp(success_data.target, targets{i}), :);
        counts = histcounts(target_data.kraus_rank, [kraus_ranks; max(kraus_ranks)+1]);
        bar(kraus_ranks + (i-1)*0.2, counts, 0.2, 'DisplayName', targets{i});
        hold on;
    end
    xlabel('Kraus Rank');
    ylabel('Count');
    title('Channel Complexity');
    legend('Location', 'best');
    grid on;
    
    % 5. Tunneling effect on dephasing
    subplot(3, 4, 5);
    b_bins = [0, 0.3, 0.7, 1.0];
    boxplot_data = {};
    group_labels = {};
    for i = 1:length(b_bins)-1
        mask = (success_data.b >= b_bins(i)) & (success_data.b < b_bins(i+1));
        data = success_data.dephasing(mask);
        data_clean = data(~isnan(data));
        if ~isempty(data_clean)
            boxplot_data{end+1} = data_clean(:);
            group_labels{end+1} = sprintf('b∈[%.1f,%.1f]', b_bins(i), b_bins(i+1));
        end
    end
    if ~isempty(boxplot_data)
        % Create grouping variable
        all_data = [];
        all_groups = [];
        for i = 1:length(boxplot_data)
            all_data = [all_data; boxplot_data{i}];
            all_groups = [all_groups; repmat(i, length(boxplot_data{i}), 1)];
        end
        boxplot(all_data, all_groups, 'Labels', group_labels);
        ylabel('Dephasing Rate');
        title('Tunneling Effect on Dephasing');
    end
    grid on;
    
    % 6. Energy asymmetry effect on dissipation
    subplot(3, 4, 6);
    a_bins = linspace(min(success_data.a), max(success_data.a), 5);
    diss_means = zeros(length(a_bins)-1, 1);
    diss_stds = zeros(length(a_bins)-1, 1);
    for i = 1:length(a_bins)-1
        mask = (success_data.a >= a_bins(i)) & (success_data.a < a_bins(i+1));
        data = success_data.dissipation(mask);
        diss_means(i) = mean(data(~isnan(data)));
        diss_stds(i) = std(data(~isnan(data)));
    end
    errorbar((a_bins(1:end-1) + a_bins(2:end))/2, diss_means, diss_stds, 'o-', 'LineWidth', 2);
    xlabel('Energy Asymmetry a');
    ylabel('Dissipation Rate');
    title('Energy Effect on Dissipation');
    grid on;
    
    % 7. Distance vs Properties
    subplot(3, 4, 7);
    scatter(success_data.dephasing, success_data.distance, 50, success_data.dissipation, 'filled');
    colorbar;
    xlabel('Dephasing');
    ylabel('Distance');
    title('Distance vs Dephasing (color=Dissipation)');
    grid on;
    
    % 8. Population asymmetry
    subplot(3, 4, 8);
    for i = 1:length(targets)
        target_data = success_data(strcmp(success_data.target, targets{i}), :);
        histogram(target_data.pop_asymmetry, 20, 'FaceAlpha', 0.5, 'DisplayName', targets{i});
        hold on;
    end
    xlabel('Population Asymmetry');
    ylabel('Count');
    title('Population Transfer');
    legend('Location', 'best');
    grid on;
    
    % 9. Time dependence
    subplot(3, 4, 9);
    t_bins = linspace(min(success_data.t), max(success_data.t), 8);
    for i = 1:length(targets)
        target_data = success_data(strcmp(success_data.target, targets{i}), :);
        success_by_t = zeros(length(t_bins)-1, 1);
        for j = 1:length(t_bins)-1
            mask = (target_data.t >= t_bins(j)) & (target_data.t < t_bins(j+1));
            success_by_t(j) = sum(mask) / max(1, length(mask));
        end
        plot((t_bins(1:end-1) + t_bins(2:end))/2, success_by_t, 'o-', 'LineWidth', 2, 'DisplayName', targets{i});
        hold on;
    end
    xlabel('Evolution Time t');
    ylabel('Success Density');
    title('Optimal Time Windows');
    legend('Location', 'best');
    grid on;
    
    % 10. Mechanism correlation matrix
    subplot(3, 4, 10);
    props = [success_data.dephasing, success_data.dissipation, ...
             success_data.coherence_loss, success_data.pop_asymmetry, ...
             success_data.unitary_weight];
    props_clean = props(all(~isnan(props), 2), :);
    if ~isempty(props_clean)
        corr_mat = corrcoef(props_clean);
        imagesc(corr_mat);
        colorbar;
        labels = {'Dephase', 'Dissip', 'Coh Loss', 'Pop Asym', 'Unitary'};
        set(gca, 'XTick', 1:5, 'XTickLabel', labels, 'XTickLabelRotation', 45);
        set(gca, 'YTick', 1:5, 'YTickLabel', labels);
        title('Property Correlations');
        colormap(gca, 'redblue');
        caxis([-1, 1]);
    end
    
    % 11. Success vs Parameter combinations
    subplot(3, 4, 11);
    param_combo = success_data.a .* success_data.t;
    scatter(param_combo, success_data.distance, 50, success_data.b, 'filled');
    colorbar;
    xlabel('a × t (Energy-Time Product)');
    ylabel('Distance');
    title('Parameter Synergy (color=b)');
    grid on;
    
    % 12. Best mechanisms by target
    subplot(3, 4, 12);
    mechanism_scores = zeros(length(targets), 4);
    for i = 1:length(targets)
        target_data = success_data(strcmp(success_data.target, targets{i}), :);
        mechanism_scores(i, 1) = mean(target_data.dephasing(~isnan(target_data.dephasing)));
        mechanism_scores(i, 2) = mean(target_data.dissipation(~isnan(target_data.dissipation)));
        mechanism_scores(i, 3) = mean(target_data.coherence_loss(~isnan(target_data.coherence_loss)));
        mechanism_scores(i, 4) = mean(target_data.pop_asymmetry(~isnan(target_data.pop_asymmetry)));
    end
    bar(mechanism_scores);
    set(gca, 'XTickLabel', targets);
    ylabel('Average Mechanism Strength');
    legend({'Dephasing', 'Dissipation', 'Coh Loss', 'Pop Asym'}, 'Location', 'best');
    title('Dominant Mechanisms by Target');
    grid on;
end

function create_mechanism_classification(results_table)
    % Classify channels by their dominant mechanism
    
    figure('Position', [150, 150, 1400, 800]);
    sgtitle('Channel Mechanism Classification', 'FontSize', 16, 'FontWeight', 'bold');
    
    success_data = results_table(results_table.success == 1, :);
    
    if isempty(success_data)
        return;
    end
    
    % Classify mechanisms
    success_data.mechanism = cell(height(success_data), 1);
    for i = 1:height(success_data)
        deph = success_data.dephasing(i);
        diss = success_data.dissipation(i);
        
        if isnan(deph) || isnan(diss)
            success_data.mechanism{i} = 'Unknown';
        elseif deph > 0.7 && diss < 0.3
            success_data.mechanism{i} = 'Pure Dephasing';
        elseif diss > 0.7 && deph < 0.3
            success_data.mechanism{i} = 'Pure Dissipation';
        elseif deph > 0.5 && diss > 0.5
            success_data.mechanism{i} = 'Mixed';
        else
            success_data.mechanism{i} = 'Weak Decoherence';
        end
    end
    
    % Plot mechanism distribution
    subplot(2, 3, 1);
    mechanisms = unique(success_data.mechanism);
    counts = zeros(length(mechanisms), 1);
    for i = 1:length(mechanisms)
        counts(i) = sum(strcmp(success_data.mechanism, mechanisms{i}));
    end
    pie(counts, mechanisms);
    title('Mechanism Distribution');
    
    % Mechanism by target
    subplot(2, 3, 2);
    targets = unique(success_data.target);
    mech_by_target = zeros(length(targets), length(mechanisms));
    for i = 1:length(targets)
        for j = 1:length(mechanisms)
            mask = strcmp(success_data.target, targets{i}) & ...
                   strcmp(success_data.mechanism, mechanisms{j});
            mech_by_target(i, j) = sum(mask);
        end
    end
    bar(mech_by_target, 'stacked');
    set(gca, 'XTickLabel', targets);
    ylabel('Count');
    legend(mechanisms, 'Location', 'best');
    title('Mechanisms by Target');
    grid on;
    
    % Parameter space by mechanism
    subplot(2, 3, 3);
    colors_mech = lines(length(mechanisms));
    for i = 1:length(mechanisms)
        mech_data = success_data(strcmp(success_data.mechanism, mechanisms{i}), :);
        scatter3(mech_data.a, mech_data.b, mech_data.t, 50, colors_mech(i,:), 'filled', ...
                'DisplayName', mechanisms{i}, 'MarkerFaceAlpha', 0.6);
        hold on;
    end
    xlabel('a'); ylabel('b'); zlabel('t');
    title('Mechanism Regions in Parameter Space');
    legend('Location', 'best');
    grid on;
    view(45, 30);
    
    % Mechanism effectiveness
    subplot(2, 3, 4);
    effectiveness = zeros(length(mechanisms), 1);
    for i = 1:length(mechanisms)
        mech_data = success_data(strcmp(success_data.mechanism, mechanisms{i}), :);
        effectiveness(i) = mean(1 - mech_data.distance);  % Lower distance = higher effectiveness
    end
    bar(effectiveness);
    set(gca, 'XTickLabel', mechanisms, 'XTickLabelRotation', 45);
    ylabel('Effectiveness (1 - distance)');
    title('Mechanism Effectiveness');
    grid on;
    
    % Dephasing vs Dissipation map with regions
    subplot(2, 3, 5);
    for i = 1:length(mechanisms)
        mech_data = success_data(strcmp(success_data.mechanism, mechanisms{i}), :);
        scatter(mech_data.dephasing, mech_data.dissipation, 100, colors_mech(i,:), 'filled', ...
                'DisplayName', mechanisms{i}, 'MarkerFaceAlpha', 0.6);
        hold on;
    end
    % Add region boundaries
    plot([0.7 0.7], [0 1], 'k--', 'LineWidth', 1);
    plot([0 1], [0.7 0.7], 'k--', 'LineWidth', 1);
    plot([0.3 0.3], [0 1], 'k:', 'LineWidth', 1);
    plot([0 1], [0.3 0.3], 'k:', 'LineWidth', 1);
    xlabel('Dephasing Rate');
    ylabel('Dissipation Rate');
    title('Mechanism Classification Map');
    legend('Location', 'best');
    grid on;
    axis([0 1 0 1]);
    
    % Distance by mechanism
    subplot(2, 3, 6);
    for i = 1:length(mechanisms)
        mech_data = success_data(strcmp(success_data.mechanism, mechanisms{i}), :);
        boxplot_data{i} = mech_data.distance;
    end
    boxplot([boxplot_data{:}], 'Labels', mechanisms);
    ylabel('Distance');
    title('Distance Distribution by Mechanism');
    set(gca, 'XTickLabelRotation', 45);
    grid on;
end

function create_optimal_channels_comparison(best_channels, discovered_channels)
    % Compare optimal channels for different targets
    
    figure('Position', [200, 200, 1400, 900]);
    sgtitle('Optimal Channel Comparison', 'FontSize', 16, 'FontWeight', 'bold');
    
    targets = fieldnames(best_channels);
    
    for i = 1:length(targets)
        target = targets{i};
        best = best_channels.(target);
        
        % Get the actual channel
        channel_key = matlab.lang.makeValidName(sprintf('a%.2f_b%.2f_t%.2f_%s', ...
            best.a, best.b, best.t, target));
        
        if isfield(discovered_channels, channel_key)
            J_Theta = discovered_channels.(channel_key);
            
            % Eigenvalue spectrum
            subplot(3, length(targets), i);
            [~, D] = eig(J_Theta);
            eigenvals = sort(real(diag(D)), 'descend');
            significant = eigenvals(eigenvals > 1e-6);
            bar(significant);
            xlabel('Index');
            ylabel('Eigenvalue');
            title(sprintf('|%s⟩: Spectrum (Rank=%d)', target, length(significant)));
            grid on;
            
            % Choi matrix visualization
            subplot(3, length(targets), length(targets) + i);
            imagesc(abs(J_Theta));
            colorbar;
            title(sprintf('|%s⟩: |J(Θ)|', target));
            axis square;
            
            % Properties radar chart
            subplot(3, length(targets), 2*length(targets) + i);
            props = [best.dephasing, best.dissipation, best.coherence_loss, ...
                    best.pop_asymmetry, best.unitary_weight];
            theta = linspace(0, 2*pi, length(props)+1);
            r = [props, props(1)];
            polarplot(theta, r, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
            thetaticks(rad2deg(theta(1:end-1)));
            thetaticklabels({'Dephase', 'Dissip', 'Coh Loss', 'Pop Asym', 'Unitary'});
            title(sprintf('|%s⟩: Properties', target));
        end
    end
end

function create_success_rate_analysis(results_table)
    % Analyze success rates across parameter space
    
    figure('Position', [250, 250, 1400, 800]);
    sgtitle('Success Rate Analysis', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Overall success rate by parameter
    subplot(2, 3, 1);
    a_unique = unique(results_table.a);
    success_by_a = arrayfun(@(x) mean(results_table.success(abs(results_table.a - x) < 1e-6)), a_unique);
    plot(a_unique, success_by_a, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Energy Asymmetry a');
    ylabel('Success Rate');
    title('Success vs Energy Asymmetry');
    grid on;
    ylim([0, 1]);
    
    subplot(2, 3, 2);
    b_unique = unique(results_table.b);
    success_by_b = arrayfun(@(x) mean(results_table.success(abs(results_table.b - x) < 1e-6)), b_unique);
    plot(b_unique, success_by_b, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Tunneling b');
    ylabel('Success Rate');
    title('Success vs Tunneling');
    grid on;
    ylim([0, 1]);
    
    subplot(2, 3, 3);
    t_unique = unique(results_table.t);
    success_by_t = arrayfun(@(x) mean(results_table.success(abs(results_table.t - x) < 1e-6)), t_unique);
    plot(t_unique, success_by_t, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Evolution Time t');
    ylabel('Success Rate');
    title('Success vs Time');
    grid on;
    ylim([0, 1]);
    
    % Success rate by target
    subplot(2, 3, 4);
    targets = unique(results_table.target);
    success_by_target = zeros(length(targets), 1);
    for i = 1:length(targets)
        success_by_target(i) = mean(results_table.success(strcmp(results_table.target, targets{i})));
    end
    bar(success_by_target);
    set(gca, 'XTickLabel', targets);
    ylabel('Success Rate');
    title('Success Rate by Target');
    grid on;
    ylim([0, 1]);
    
    % Combined parameter effects
    subplot(2, 3, 5);
    % Create bins for combined analysis
    a_bins = quantile(results_table.a, [0, 0.33, 0.67, 1]);
    b_bins = quantile(results_table.b, [0, 0.33, 0.67, 1]);
    
    success_matrix = zeros(3, 3);
    for i = 1:3
        for j = 1:3
            mask = (results_table.a >= a_bins(i) & results_table.a < a_bins(i+1)) & ...
                   (results_table.b >= b_bins(j) & results_table.b < b_bins(j+1));
            if sum(mask) > 0
                success_matrix(i, j) = mean(results_table.success(mask));
            end
        end
    end
    
    imagesc(success_matrix);
    colorbar;
    xlabel('Tunneling (Low/Med/High)');
    ylabel('Energy Asymmetry (Low/Med/High)');
    title('Combined Parameter Effects');
    set(gca, 'XTick', 1:3, 'XTickLabel', {'Low', 'Med', 'High'});
    set(gca, 'YTick', 1:3, 'YTickLabel', {'Low', 'Med', 'High'});
    colormap(gca, flipud(hot));
    
    % Distance distribution for successful cases
    subplot(2, 3, 6);
    success_data = results_table(results_table.success == 1, :);
    for i = 1:length(targets)
        target_data = success_data(strcmp(success_data.target, targets{i}), :);
        histogram(target_data.distance, 20, 'FaceAlpha', 0.5, 'DisplayName', targets{i});
        hold on;
    end
    xlabel('Distance');
    ylabel('Count');
    title('Distance Distribution (Successful)');
    legend('Location', 'best');
    grid on;
end

function create_phase_diagrams(results_table, a_values, b_values, t_values)
    % Create phase diagrams showing regions of success
    
    figure('Position', [300, 300, 1600, 1000]);
    sgtitle('Phase Diagrams: Regions of Successful Conversion', 'FontSize', 16, 'FontWeight', 'bold');
    
    targets = unique(results_table.target);
    
    % For each target, create phase diagrams
    for i_target = 1:length(targets)
        target = targets{i_target};
        target_data = results_table(strcmp(results_table.target, target), :);
        
        % a-b phase diagram (averaged over t)
        subplot(2, 3, i_target);
        phase_ab = zeros(length(a_values), length(b_values));
        for i = 1:length(a_values)
            for j = 1:length(b_values)
                mask = (abs(target_data.a - a_values(i)) < 1e-6) & ...
                       (abs(target_data.b - b_values(j)) < 1e-6);
                if sum(mask) > 0
                    % Use average distance as metric
                    distances = target_data.distance(mask);
                    phase_ab(i, j) = mean(distances(~isnan(distances)));
                else
                    phase_ab(i, j) = 2;
                end
            end
        end
        
        % Plot with contour lines
        imagesc(b_values, a_values, phase_ab);
        hold on;
        contour(b_values, a_values, phase_ab, [0.05, 0.1, 0.5], 'k-', 'LineWidth', 2);
        colorbar;
        xlabel('Tunneling b');
        ylabel('Energy asymmetry a');
        title(sprintf('Target |%s⟩: a-b Phase', target));
        set(gca, 'YDir', 'normal');
        colormap(gca, hot);
        caxis([0, 2]);
        
        % a-t phase diagram (averaged over b)
        subplot(2, 3, 3 + i_target);
        phase_at = zeros(length(a_values), length(t_values));
        for i = 1:length(a_values)
            for j = 1:length(t_values)
                mask = (abs(target_data.a - a_values(i)) < 1e-6) & ...
                       (abs(target_data.t - t_values(j)) < 1e-6);
                if sum(mask) > 0
                    distances = target_data.distance(mask);
                    phase_at(i, j) = mean(distances(~isnan(distances)));
                else
                    phase_at(i, j) = 2;
                end
            end
        end
        
        imagesc(t_values, a_values, phase_at);
        hold on;
        contour(t_values, a_values, phase_at, [0.05, 0.1, 0.5], 'k-', 'LineWidth', 2);
        colorbar;
        xlabel('Time t');
        ylabel('Energy asymmetry a');
        title(sprintf('Target |%s⟩: a-t Phase', target));
        set(gca, 'YDir', 'normal');
        colormap(gca, hot);
        caxis([0, 2]);
    end
end

function print_summary_statistics(results_table, best_channels)
    % Print comprehensive summary statistics
    
    fprintf('\n');
    fprintf('=========================================\n');
    fprintf('SUMMARY STATISTICS\n');
    fprintf('=========================================\n\n');
    
    % Overall statistics
    total_sims = height(results_table);
    total_success = sum(results_table.success);
    success_rate = 100 * total_success / total_sims;
    
    fprintf('📊 OVERALL PERFORMANCE:\n');
    fprintf('  Total simulations: %d\n', total_sims);
    fprintf('  Successful conversions: %d\n', total_success);
    fprintf('  Success rate: %.1f%%\n', success_rate);
    fprintf('  Average distance (all): %.4f\n', mean(results_table.distance));
    fprintf('  Average distance (successful): %.6f\n', mean(results_table.distance(results_table.success == 1)));
    
    % By target
    fprintf('\n🎯 PERFORMANCE BY TARGET:\n');
    targets = unique(results_table.target);
    for i = 1:length(targets)
        target = targets{i};
        target_data = results_table(strcmp(results_table.target, target), :);
        target_success = sum(target_data.success);
        target_rate = 100 * target_success / height(target_data);
        
        fprintf('  Target |%s⟩:\n', target);
        fprintf('    Success rate: %.1f%% (%d/%d)\n', target_rate, target_success, height(target_data));
        
        if target_success > 0
            successful = target_data(target_data.success == 1, :);
            fprintf('    Avg distance: %.6f\n', mean(successful.distance));
            fprintf('    Avg dephasing: %.4f\n', mean(successful.dephasing(~isnan(successful.dephasing))));
            fprintf('    Avg dissipation: %.4f\n', mean(successful.dissipation(~isnan(successful.dissipation))));
        end
    end
    
    % Best channels
    fprintf('\n🏆 OPTIMAL CHANNELS:\n');
    target_names = fieldnames(best_channels);
    for i = 1:length(target_names)
        target = target_names{i};
        best = best_channels.(target);
        
        fprintf('  Target |%s⟩:\n', target);
        fprintf('    Parameters: a=%.2f, b=%.2f, t=%.2f\n', best.a, best.b, best.t);
        fprintf('    Distance: %.6f\n', best.distance);
        fprintf('    Mechanism:\n');
        fprintf('      Dephasing: %.4f\n', best.dephasing);
        fprintf('      Dissipation: %.4f\n', best.dissipation);
        fprintf('      Coherence loss: %.4f\n', best.coherence_loss);
        fprintf('      Population asymmetry: %.4f\n', best.pop_asymmetry);
        fprintf('      Unitary weight: %.4f\n', best.unitary_weight);
        fprintf('      Kraus rank: %d\n', best.kraus_rank);
        
        % Classify mechanism
        if best.dephasing > 0.7 && best.dissipation < 0.3
            mechanism = 'PURE DEPHASING';
        elseif best.dissipation > 0.7 && best.dephasing < 0.3
            mechanism = 'PURE DISSIPATION';
        elseif best.dephasing > 0.5 && best.dissipation > 0.5
            mechanism = 'MIXED DECOHERENCE';
        else
            mechanism = 'WEAK/COMPLEX';
        end
        fprintf('      Classification: %s\n', mechanism);
    end
    
    % Mechanism analysis
    fprintf('\n🔬 MECHANISM ANALYSIS:\n');
    success_data = results_table(results_table.success == 1, :);
    
    if ~isempty(success_data)
        fprintf('  Dephasing-dominated: %.1f%% (rate > 0.7)\n', ...
            100 * sum(success_data.dephasing > 0.7 & ~isnan(success_data.dephasing)) / height(success_data));
        fprintf('  Dissipation-dominated: %.1f%% (rate > 0.7)\n', ...
            100 * sum(success_data.dissipation > 0.7 & ~isnan(success_data.dissipation)) / height(success_data));
        fprintf('  Mixed mechanisms: %.1f%% (both > 0.5)\n', ...
            100 * sum(success_data.dephasing > 0.5 & success_data.dissipation > 0.5 & ...
                     ~isnan(success_data.dephasing) & ~isnan(success_data.dissipation)) / height(success_data));
        fprintf('  High unitarity: %.1f%% (weight > 0.7)\n', ...
            100 * sum(success_data.unitary_weight > 0.7 & ~isnan(success_data.unitary_weight)) / height(success_data));
    end
    
    % Parameter insights
    fprintf('\n💡 PARAMETER INSIGHTS:\n');
    
    % Find optimal ranges
    if ~isempty(success_data)
        fprintf('  Optimal energy asymmetry: a ∈ [%.2f, %.2f]\n', ...
            quantile(success_data.a, 0.25), quantile(success_data.a, 0.75));
        fprintf('  Optimal tunneling: b ∈ [%.2f, %.2f]\n', ...
            quantile(success_data.b, 0.25), quantile(success_data.b, 0.75));
        fprintf('  Optimal evolution time: t ∈ [%.2f, %.2f]\n', ...
            quantile(success_data.t, 0.25), quantile(success_data.t, 0.75));
        
        % Correlations
        if height(success_data) > 2
            fprintf('\n  Parameter correlations with success:\n');
            a_corr = corr(results_table.a, double(results_table.success));
            b_corr = corr(results_table.b, double(results_table.success));
            t_corr = corr(results_table.t, double(results_table.success));
            fprintf('    Energy asymmetry (a): %.3f\n', a_corr);
            fprintf('    Tunneling (b): %.3f\n', b_corr);
            fprintf('    Time (t): %.3f\n', t_corr);
        end
    end
    
    % Practical recommendations
    fprintf('\n🎓 PRACTICAL RECOMMENDATIONS:\n');
    fprintf('  1. For reliable homochirality (distance < 0.05):\n');
    if ~isempty(success_data)
        excellent = success_data(success_data.distance < 0.01, :);
        if ~isempty(excellent)
            fprintf('     • Use a ≈ %.2f ± %.2f\n', mean(excellent.a), std(excellent.a));
            fprintf('     • Use b ≈ %.2f ± %.2f\n', mean(excellent.b), std(excellent.b));
            fprintf('     • Use t ≈ %.2f ± %.2f\n', mean(excellent.t), std(excellent.t));
        end
        
        fprintf('  2. Dominant mechanism required:\n');
        avg_deph = mean(success_data.dephasing(~isnan(success_data.dephasing)));
        avg_diss = mean(success_data.dissipation(~isnan(success_data.dissipation)));
        if avg_deph > avg_diss
            fprintf('     • DEPHASING (avg rate: %.2f)\n', avg_deph);
            fprintf('     • Destroys quantum coherences between L and R\n');
        else
            fprintf('     • DISSIPATION (avg rate: %.2f)\n', avg_diss);
            fprintf('     • Actively transfers population between L and R\n');
        end
        
        fprintf('  3. Channel complexity:\n');
        avg_rank = mean(success_data.kraus_rank(~isnan(success_data.kraus_rank)));
        fprintf('     • Average Kraus rank: %.1f\n', avg_rank);
        if avg_rank < 2.5
            fprintf('     • Simple channels suffice (nearly unitary)\n');
        else
            fprintf('     • Complex multi-operator channels needed\n');
        end
    end
    
    fprintf('\n=========================================\n');
end

%% HELPER FUNCTIONS

function [distance, J_Theta, status] = solve_conversion_sdp_silent(J_Phi, J_Psi)
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
    status = cvx_status;
end

function analysis = analyze_discovered_channel(J_Theta, J_Phi)
    % Comprehensive analysis of discovered superchannel
    
    % Basic properties
    [V, D] = eig(J_Theta);
    eigenvalues = sort(real(diag(D)), 'descend');
    
    % Kraus rank (number of significant eigenvalues)
    threshold = 1e-6;
    kraus_rank = sum(eigenvalues > threshold);
    
    % Decompose into blocks to analyze mechanism
    dim = size(J_Theta, 1);
    n = sqrt(dim);  % Should be 4 for qubit channels
    
    J_reshaped = reshape(J_Theta, [n, n, n, n]);
    
    % Dephasing: loss of off-diagonal coherences
    diagonal_norm = 0;
    offdiag_norm = 0;
    for i = 1:n
        for j = 1:n
            block = squeeze(J_reshaped(i, j, :, :));
            block_norm = norm(block, 'fro')^2;
            if i == j
                diagonal_norm = diagonal_norm + block_norm;
            else
                offdiag_norm = offdiag_norm + block_norm;
            end
        end
    end
    
    total_norm = diagonal_norm + offdiag_norm;
    dephasing_rate = 1 - offdiag_norm / max(total_norm, 1e-10);
    
    % Dissipation: population transfer asymmetry
    pop_00_norm = norm(squeeze(J_reshaped(1, 1, :, :)), 'fro');
    pop_11_norm = norm(squeeze(J_reshaped(2, 2, :, :)), 'fro');
    dissipation_rate = abs(pop_00_norm - pop_11_norm) / max(pop_00_norm + pop_11_norm, 1e-10);
    
    % Coherence loss: compare input vs output coherence preservation
    coherence_blocks = [squeeze(J_reshaped(1, 2, :, :)), squeeze(J_reshaped(2, 1, :, :))];
    coherence_loss = 1 - norm(coherence_blocks, 'fro') / sqrt(2 * n^2);
    
    % Population asymmetry
    population_asymmetry = abs(pop_00_norm - pop_11_norm) / (pop_00_norm + pop_11_norm + 1e-10);
    
    % Unitary weight
    trace_Theta = trace(J_Theta);
    unitary_weight = eigenvalues(1) / max(trace_Theta, 1e-10);
    
    % Package results
    analysis = struct();
    analysis.eigenvalues = eigenvalues;
    analysis.kraus_rank = kraus_rank;
    analysis.dephasing_rate = dephasing_rate;
    analysis.dissipation_rate = dissipation_rate;
    analysis.coherence_loss = coherence_loss;
    analysis.population_asymmetry = population_asymmetry;
    analysis.unitary_weight = unitary_weight;
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