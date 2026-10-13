function run_wesad_loso_ml_only()

% =========================================================================
% WESAD - Binary LOSO (ACT vs. INH)
%
% This version assumes that all feature extraction has already been
% performed and the feature tables are available in a MAT file.
%
% The script:
%   - Loads ECG-only and ECG+RESP feature tables
%   - Performs LOSO evaluation
%   - Trains four classifiers:
%       * LDA
%       * SVM (RBF)
%       * KNN
%       * Random Forest
%   - Computes Accuracy, F1-score, Sensitivity,
%     Specificity, Balanced Accuracy and AUC
% =========================================================================

clc;
close all;
clear all;
rng(1,'twister');

t_start = tic;

%% =========================================================
% Configuration
%% =========================================================

CFG = struct();

CFG.balance_method  = 'Weights';
CFG.use_fs          = false;
CFG.fs_num_features = 10;

CFG.verbose = true;

CFG.out_root = fullfile( ...
    pwd, ...
    ['EXP4_results_ML_only_' datestr(now,'yyyy-mm-dd_HHMMSS')]);

if ~isfolder(CFG.out_root)
    mkdir(CFG.out_root);
end

%% =========================================================
% Diary
%% =========================================================

diary_file = fullfile(CFG.out_root,'results.txt');

if isfile(diary_file)
    delete(diary_file);
end

diary(diary_file);

fprintf('========================================================\n');
fprintf('Execution Date: %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'));
fprintf('Balancing Method: %s\n', CFG.balance_method);
fprintf('========================================================\n\n');

%% =========================================================
% Load feature tables
%% =========================================================

S = load('EXP4_features.mat');

All_ECR = S.All_ECR;

fprintf('Feature tables loaded successfully.\n\n');

%% =========================================================
% Subject summary
%% =========================================================

print_subject_summary(All_ECR,'ECG+RESP');

%% =========================================================
% Classifiers
%% =========================================================

CLS.LDA.DiscrimType = 'pseudoLinear';
CLS.SVM.Kernel      = 'rbf';
CLS.KNN.K           = 5;
CLS.RF.Trees        = 250;

%% =========================================================
% EXPERIMENT 4: ECG + RESP experiment (27 features)
%% =========================================================

[R_ecr, fs_ecr] = loso_eval_bin(All_ECR,true,CFG,CLS,CFG.out_root);

%% =========================================================
% Global results
%% =========================================================

print_global_summary('ECG+RESP', R_ecr);

if CFG.use_fs
    n_folds_ecg = numel(unique(All_ECG.Subject));
    print_feature_selection_summary('ECG-only',fs_ecg,n_folds_ecg);
    n_folds_ecr = numel(unique(All_ECR.Subject));
    print_feature_selection_summary('ECG+RESP',fs_ecr,n_folds_ecr);

end

if CFG.verbose
    elapsed = toc(t_start);
    fprintf('\nExecution time: %.2f seconds (%.2f minutes)\n',elapsed,elapsed/60);
    fprintf('Results stored in: %s\n', CFG.out_root);
end

diary off;

end

% ========================================================================
% Helper functions
% ========================================================================

function print_subject_summary(T, tag)

    if isempty(T)
        fprintf('(%s) Empty table.\n', tag);
        return
    end

    subs = unique(T.Subject,'stable');

    tot = height(T);

    n1 = sum(T.y==1);
    n0 = sum(T.y==0);

    fprintf('--- Subject summary (%s) ---\n', tag);

    fprintf('Total windows: %d | Class 0 = %d (%.1f%%) | Class 1 = %d (%.1f%%)\n', ...
        tot, n0, 100*n0/tot, n1, 100*n1/tot);

    for i = 1:numel(subs)

        si = strcmp(T.Subject, subs{i});

        n = sum(si);

        n1s = sum(T.y(si)==1);
        n0s = sum(T.y(si)==0);

        fprintf('%s: n=%d | Class 0 = %d | Class 1 = %d\n', ...
            subs{i}, n, n0s, n1s);

    end

    fprintf('\n');

end

function [R, all_selected_feats] = loso_eval_bin(T, with_resp, CFG, CLS, outdir)

    R = table();
    all_selected_feats = {};

    if isempty(T)
        warning('Empty table in loso_eval_bin.');
        return
    end

    subs = unique(T.Subject,'stable');

    for i = 1:numel(subs)

        testSub = subs{i};

        Te = T(strcmp(T.Subject,testSub), :);
        Tr = T(~strcmp(T.Subject,testSub), :);

        ytr = Tr.y;
        yte = Te.y;

        Xtr = Tr{:, setdiff(Tr.Properties.VariableNames,{'Subject','y'})};
        Xte = Te{:, setdiff(Te.Properties.VariableNames,{'Subject','y'})};

        [Xtr, Xte, kept] = prune_predictors(Xtr, Xte);

        if isempty(Xtr) || isempty(Xte)

            warning('[%s] No predictors after pruning.', testSub);
            continue

        end

        XtrS = Xtr;
        XteS = Xte;

        c0_pre = sum(ytr==0);
        c1_pre = sum(ytr==1);

        if strcmpi(CFG.balance_method,'SMOTE')

            [XtrS,ytr] = apply_smote(XtrS,ytr,5);

            W = ones(size(ytr));

        elseif strcmpi(CFG.balance_method,'RUS')

            [XtrS,ytr] = apply_rus(XtrS,ytr);

            W = ones(size(ytr));

        else

            p1 = mean(ytr==1);
            p0 = mean(ytr==0);

            if p1==0 || p0==0

                W = ones(size(ytr));

            else

                W = zeros(size(ytr));

                W(ytr==0) = 0.5/p0;
                W(ytr==1) = 0.5/p1;

            end

        end

        c0_post = sum(ytr==0);
        c1_post = sum(ytr==1);

        fig_bal_dir = fullfile(outdir,'figs_balance');

        if ~isfolder(fig_bal_dir)
            mkdir(fig_bal_dir);
        end

        h = figure('Visible','off');

        bar([c0_pre,c1_pre; c0_post,c1_post]','grouped');

        set(gca,'XTickLabel',{'Class 0','Class 1'});

        legend('Before','After','Location','northeast');

        title(sprintf('[%s] Class balancing (%s)', ...
            testSub, CFG.balance_method));

        ylabel('Number of windows');
        tag_mod = ternary(with_resp,'ECG+RESP','ECG-only');
        saveas(h,fullfile(fig_bal_dir,sprintf('%s_%s_balance.png',testSub,tag_mod)));
        close(h);
        feature_names = T.Properties.VariableNames(3:end);
        feature_names = feature_names(kept);

        if CFG.use_fs

            n_sel = min(CFG.fs_num_features,size(XtrS,2));

            [idx_ranks,~] = relieff(XtrS,ytr,10);

            sel_idx = idx_ranks(1:n_sel);

            XtrS = XtrS(:,sel_idx);
            XteS = XteS(:,sel_idx);

            all_selected_feats = ...
                [all_selected_feats feature_names(sel_idx)];

        end

        M = train_models(XtrS,ytr,W,CLS);

        [ypL,ypS,ypK,ypR] = deal([]);

        scoreL = [];
        scoreS = [];
        scoreK = [];
        scoreR = [];

        if ~isempty(M.LDA)
            [ypL,scoreL] = predict(M.LDA,XteS);
        end

        if ~isempty(M.SVM)
            [ypS,scoreS] = predict(M.SVM,XteS);
        end

        if ~isempty(M.KNN)
            [ypK,scoreK] = predict(M.KNN,XteS);
        end

        if ~isempty(M.RF)
            [ypR_,scoreR] = predict(M.RF,XteS);
            ypR = str2double(ypR_);
        end

        [accL,f1L,sensL,specL,balaccL] = f1score(yte,ypL);
        [accS,f1S,sensS,specS,balaccS] = f1score(yte,ypS);
        [accK,f1K,sensK,specK,balaccK] = f1score(yte,ypK);
        [accR,f1R,sensR,specR,balaccR] = f1score(yte,ypR);

        aucL = calc_auc(yte,scoreL(:,2));
        aucS = calc_auc(yte,scoreS(:,2));
        aucK = calc_auc(yte,scoreK(:,2));
        aucR = calc_auc(yte,scoreR(:,2));

        R = [R;
            table( ...
            {testSub}, ...
            accL,accS,accK,accR, ...
            f1L,f1S,f1K,f1R, ...
            sensL,sensS,sensK,sensR, ...
            specL,specS,specK,specR, ...
            balaccL,balaccS,balaccK,balaccR, ...
            aucL,aucS,aucK,aucR, ...
            sum(kept), ...
            'VariableNames',{ ...
            'Subject', ...
            'Acc_LDA','Acc_SVM','Acc_KNN','Acc_RF', ...
            'F1_LDA','F1_SVM','F1_KNN','F1_RF', ...
            'Sens_LDA','Sens_SVM','Sens_KNN','Sens_RF', ...
            'Spec_LDA','Spec_SVM','Spec_KNN','Spec_RF', ...
            'Balacc_LDA','Balacc_SVM','Balacc_KNN','Balacc_RF', ...
            'AUC_LDA','AUC_SVM','AUC_KNN','AUC_RF', ...
            'nfeats'})];

        fprintf( ...
            '[%s] -> Acc (L/S/K/R)= %.1f/%.1f/%.1f/%.1f | F1 (L/S/K/R)= %.1f/%.1f/%.1f/%.1f | feats=%d\n', ...
            testSub, 100*[accL accS accK accR], 100*[f1L f1S f1K f1R], sum(kept));

    end

    tag = ternary(with_resp,'ECG+RESP','ECG-only');

    writetable(R,fullfile(outdir,['results_loso_' tag '.csv']));

end

function [Xtr2, Xte2, kept] = prune_predictors(Xtr, Xte)

    good = all(isfinite(Xtr),1) & (std(Xtr,0,1) > 0);

    Xtr2 = Xtr(:,good);
    Xte2 = Xte(:,good);

    kept = good;

end

function M = train_models(Xtr, ytr, W, CLS)

    M = struct('LDA',[],'SVM',[],'KNN',[],'RF',[]);

    try
        M.LDA = fitcdiscr(Xtr, ytr,'DiscrimType',CLS.LDA.DiscrimType, ...
            'Prior','empirical', 'Weights',W);
    catch
        M.LDA = [];
    end

    try
        M.SVM = fitcsvm(Xtr, ytr, 'KernelFunction',CLS.SVM.Kernel, ...
            'KernelScale','auto', 'Standardize',false, ...
            'Weights',W, 'ClassNames',[0 1]);
    catch
        M.SVM = [];
    end

    try
        M.KNN = fitcknn(Xtr, ytr, 'NumNeighbors',CLS.KNN.K, ...
            'Standardize',false, 'Weights',W);
    catch
        M.KNN = [];
    end

    try
        M.RF = TreeBagger(CLS.RF.Trees, Xtr, ytr, 'Method','classification', ...
            'OOBPrediction','off','Weights',W,'MinLeafSize',5, ...
            'NumPredictorsToSample','all');
    catch
        M.RF = [];
    end

end

function [acc, f1, sens, spec, balacc] = f1score(ytrue, ypred)

    if isempty(ypred)

        acc = NaN;
        f1 = NaN;
        sens = NaN;
        spec = NaN;
        balacc = NaN;

        return

    end

    ytrue = ytrue(:);
    ypred = ypred(:);

    tp = sum((ytrue==1) & (ypred==1));
    tn = sum((ytrue==0) & (ypred==0));
    fp = sum((ytrue==0) & (ypred==1));
    fn = sum((ytrue==1) & (ypred==0));

    total = tp + tn + fp + fn;

    if total==0
        acc = NaN;
    else
        acc = (tp + tn) / total;
    end

    if (tp+fp)==0
        prec = 0;
    else
        prec = tp/(tp+fp);
    end

    if (tp+fn)==0
        sens = 0;
    else
        sens = tp/(tp+fn);
    end

    if (tn+fp)==0
        spec = 0;
    else
        spec = tn/(tn+fp);
    end

    if (prec+sens)==0
        f1 = 0;
    else
        f1 = 2*prec*sens/(prec+sens);
    end

    balacc = (sens + spec)/2;

end

function auc = calc_auc(ytrue, score)

    try
        [~,~,~,auc] = perfcurve(ytrue,score,1);
    catch
        auc = NaN;
    end

end

function y = ternary(cond,a,b)

    if cond
        y = a;
    else
        y = b;
    end

end

function print_global_summary(tag, R)

    if isempty(R)

        fprintf('\n%s : NO RESULTS AVAILABLE\n', tag);
        return

    end

    accL = mean(R.Acc_LDA,'omitnan');
    accS = mean(R.Acc_SVM,'omitnan');
    accK = mean(R.Acc_KNN,'omitnan');
    accR = mean(R.Acc_RF ,'omitnan');

    f1L = mean(R.F1_LDA,'omitnan');
    f1S = mean(R.F1_SVM,'omitnan');
    f1K = mean(R.F1_KNN,'omitnan');
    f1R = mean(R.F1_RF ,'omitnan');

    sensL = mean(R.Sens_LDA,'omitnan');
    sensS = mean(R.Sens_SVM,'omitnan');
    sensK = mean(R.Sens_KNN,'omitnan');
    sensR = mean(R.Sens_RF ,'omitnan');

    specL = mean(R.Spec_LDA,'omitnan');
    specS = mean(R.Spec_SVM,'omitnan');
    specK = mean(R.Spec_KNN,'omitnan');
    specR = mean(R.Spec_RF ,'omitnan');

    baccL = mean(R.Balacc_LDA,'omitnan');
    baccS = mean(R.Balacc_SVM,'omitnan');
    baccK = mean(R.Balacc_KNN,'omitnan');
    baccR = mean(R.Balacc_RF ,'omitnan');

    aucL = mean(R.AUC_LDA,'omitnan');
    aucS = mean(R.AUC_SVM,'omitnan');
    aucK = mean(R.AUC_KNN,'omitnan');
    aucR = mean(R.AUC_RF ,'omitnan');

    fprintf('\n=========================================\n');
    fprintf('%s\n', tag);
    fprintf('=========================================\n');

    fprintf('LDA: Acc=%.2f%% F1=%.2f%% Sens=%.2f%% Spec=%.2f%% BAcc=%.2f%% AUC=%.3f\n', ...
        100*accL,100*f1L,100*sensL,100*specL,100*baccL,aucL);

    fprintf('SVM: Acc=%.2f%% F1=%.2f%% Sens=%.2f%% Spec=%.2f%% BAcc=%.2f%% AUC=%.3f\n', ...
        100*accS,100*f1S,100*sensS,100*specS,100*baccS,aucS);

    fprintf('KNN: Acc=%.2f%% F1=%.2f%% Sens=%.2f%% Spec=%.2f%% BAcc=%.2f%% AUC=%.3f\n', ...
        100*accK,100*f1K,100*sensK,100*specK,100*baccK,aucK);

    fprintf('RF : Acc=%.2f%% F1=%.2f%% Sens=%.2f%% Spec=%.2f%% BAcc=%.2f%% AUC=%.3f\n', ...
        100*accR,100*f1R,100*sensR,100*specR,100*baccR,aucR);

end
