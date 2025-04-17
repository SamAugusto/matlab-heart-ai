%% Heart Disease Analysis
% Load the dataset
Heart_Data = readtable("C:\Users\Samuel\Desktop\Python Projects\Matalab Projects\Heart Disease Project\heart.csv");
Cardio_data = readtable("C:\Users\Samuel\Desktop\Python Projects\Matalab Projects\Heart Disease Project\cardio_train.csv");
% Separate the data based on target (0 = No disease, 1 = Disease)
target_0 = Heart_Data(Heart_Data.target == 0, :);
target_1 = Heart_Data(Heart_Data.target == 1, :);

%% Create Age Distribution Plots
figure(1);

% Subplot 1: Age Distribution for Patients with Heart Disease
subplot(3, 1, 1);
histogram(target_1.age, 'BinMethod', 'auto', 'FaceColor', 'r', 'EdgeColor', 'k');
xlabel('Age');
ylabel('Count');
title('Age Distribution for Patients with Heart Disease');
legend('Disease (1 = Yes)');
% Subplot 2: Age Distribution for Patients without Heart Disease
subplot(3, 1, 2);
histogram(target_0.age, 'BinMethod', 'auto', 'FaceColor', 'b', 'EdgeColor', 'k');
xlabel('Age');
ylabel('Count');
title('Age Distribution for Patients without Heart Disease');
legend('No Disease (0 = No)');

% Subplot 3: Age Distribution for All Patients
subplot(3, 1, 3);
histogram(Heart_Data.age, 'BinMethod', 'auto', 'FaceColor', 'g', 'EdgeColor', 'k');
xlabel('Age');
ylabel('Count');
title('Age Distribution for All Patients');
legend('All Patients');

%% Statistical Tests
% Null Hypothesis:
% There is no significant difference between the age distributions of patients with and without heart disease.

% Variance test (F-test)
[~, p_var] = vartest2(target_0.age, target_1.age);
if p_var < 0.05
    disp('Variances are significantly different. Using Welch''s t-test and Mann-Whitney U test.');
    % Welch's t-test
    [h_ttest, p_ttest, ci_ttest, stats_ttest] = ttest2(target_0.age, target_1.age, 'Vartype', 'unequal');
    % Mann-Whitney U test
    [p_u, h_u] = ranksum(target_0.age, target_1.age);
else
    disp('Variances are similar. Using standard t-test.');
    % Standard t-test
    [h_ttest, p_ttest, ci_ttest, stats_ttest] = ttest2(target_0.age, target_1.age);
end

% Display t-test results
if h_ttest == 0
    disp('Fail to reject the null hypothesis: No significant difference in age distribution.');
else
    disp('Reject the null hypothesis: Significant difference in age distribution.');
end
fprintf('p-value (t-test): %.4f\n', p_ttest);
fprintf('Confidence Interval (t-test): [%.4f, %.4f]\n', ci_ttest(1), ci_ttest(2));
fprintf('t-statistic: %.4f\n', stats_ttest.tstat);

% Display Mann-Whitney U test results
if h_u == 0
    disp('Mann-Whitney: Fail to reject the null hypothesis.');
else
    disp('Mann-Whitney: Reject the null hypothesis.');
end
fprintf('p-value (Mann-Whitney): %.4f\n', p_u);

%% Comparisons with Total Population
% Variance test between Heart_Data.age and target_1.age
[~, p_var_total] = vartest2(Heart_Data.age, target_1.age);
if p_var_total < 0.05
    disp('Variances are significantly different (Total Population vs Heart Disease). Using Welch''s t-test.');
    [h_ttest_total, p_ttest_total, ci_ttest_total, stats_ttest_total] = ttest2(Heart_Data.age, target_1.age, 'Vartype', 'unequal');
else
    disp('Variances are similar. Using standard t-test.');
    [h_ttest_total, p_ttest_total, ci_ttest_total, stats_ttest_total] = ttest2(Heart_Data.age, target_1.age);
end

% Display results for total population comparison
if h_ttest_total == 0
    disp('Fail to reject the null hypothesis: No significant difference (Total vs Heart Disease).');
else
    disp('Reject the null hypothesis: Significant difference (Total vs Heart Disease).');
end
fprintf('p-value (t-test): %.4f\n', p_ttest_total);
fprintf('Confidence Interval (t-test): [%.4f, %.4f]\n', ci_ttest_total(1), ci_ttest_total(2));

%% Gender Distribution Analysis
% Bar plot for gender distribution
figure(2);

% Overall gender distribution
subplot(3, 1, 1);
gender_counts = [sum(Heart_Data.sex == 0), sum(Heart_Data.sex == 1)];
bar(categorical({'Female', 'Male'}), gender_counts, 'FaceColor', 'c');
title('Overall Gender Distribution');
ylabel('Count');

% Gender distribution in patients with heart disease
subplot(3, 1, 2);
gender_counts_disease = [sum(target_1.sex == 0), sum(target_1.sex == 1)];
bar(categorical({'Female', 'Male'}), gender_counts_disease, 'FaceColor', 'm');
title('Gender Distribution (Heart Disease)');
ylabel('Count');

% Gender distribution in patients without heart disease
subplot(3, 1, 3);
gender_counts_no_disease = [sum(target_0.sex == 0), sum(target_0.sex == 1)];
bar(categorical({'Female', 'Male'}), gender_counts_no_disease, 'FaceColor', 'y');
title('Gender Distribution (No Heart Disease)');
ylabel('Count');

%% Heart Disease Analysis
% Load the dataset
Heart_Data = readtable("C:\Users\Samuel\Desktop\Python Projects\Matalab Projects\Heart Disease Project\heart.csv");

% Separate the data based on target (0 = No disease, 1 = Disease)
target_0 = Heart_Data(Heart_Data.target == 0, :);
target_1 = Heart_Data(Heart_Data.target == 1, :);

%% Machine learning data
t = Heart_Data.target; % target variable
Icross = crossvalind('kfold', size(Heart_Data, 1), 8); % Assign fold indices to each patient.
Itest = Icross == 1;
Itrain = ~Itest;

net = fitnet(200);
% Split data into training and testing sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% training, 20% testing

Xtrain = X(training(cv), :);
ytrain = y(training(cv), :);
Xtest = X(test(cv), :);
ytest = y(test(cv), :);

% Create and train the neural network
net = fitnet(10); % 10 hidden units (can be tuned)
net.trainParam.showWindow = false; % Suppress GUI

% Set data division (training/validation)
net.divideParam.trainRatio = 0.9;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0; % Test data is separate

% Train the network
net = train(net, Xtrain', ytrain'); % Transpose for neural network input

% Test the network
% Predict on training and testing data
ytrain_pred = net(Xtrain')';
ytest_pred = net(Xtest')';

% Round predictions to 0 or 1 (binary classification)
ytrain_pred_rounded = round(ytrain_pred);
ytest_pred_rounded = round(ytest_pred);

%Evaluate performance
% Calculate accuracy
train_accuracy = sum(ytrain_pred_rounded == ytrain) / numel(ytrain) * 100;
test_accuracy = sum(ytest_pred_rounded == ytest) / numel(ytest) * 100;

fprintf('Training Accuracy: %.2f%%\n', train_accuracy);
fprintf('Testing Accuracy: %.2f%%\n', test_accuracy);

% Confusion matrix for test data
confusionchart(ytest, ytest_pred_rounded);
title('Confusion Matrix for Test Data');

% Plot error histogram
figure;
errors = ytest_pred - ytest;
ploterrhist(errors);
title('Error Histogram');

%% Run the prediction model
%save('HeartDiseaseModel.mat', 'net');
% Load the trained model
load('UpdatedHeartDiseaseModel.mat', 'net');

% Predict using new data
heart_data = readtable('cardio_train.csv');

% Remove the target column if present
if any(strcmp(heart_data.Properties.VariableNames, 'target'))
    heart_data(:, 'target') = [];
end

% Convert the table to a matrix
input_data = table2array(heart_data);

% Run predictions
predictions = sim(net, input_data');

% Graph how many people will have heart disease (prediction vs. actual)
figure;

% Subplot 1: Predictions Distribution
subplot(2,1,1);
histogram(predictions, 'BinEdges', [-0.5, 0.5, 1.5], 'FaceColor', 'y');
xlabel('Predicted Values (0 = No Disease, 1 = Disease)');
ylabel('Count');
title('Prediction Distribution');
grid on;

% Subplot 2: Actual Target Distribution
subplot(2,1,2);
histogram(Cardio_data.cardio, 'BinEdges', [-0.5, 0.5, 1.5], 'FaceColor', 'k');
xlabel('Actual Values (0 = No Disease, 1 = Disease)');
ylabel('Count');
title('Actual Distribution');
grid on;

%%More training data

%% Incremental Learning: Fine-tune the existing model with new data
net.trainParam.lr = 0.001;  % Set the learning rate to a low value
net = fitnet(10);

% Load the pre-trained model (assuming you have already trained and saved it)
load('HeartDiseaseModel.mat', 'net');

% Load the new data for prediction or retraining
heart_data = readtable('cardio_train.csv');

% Remove the target column if present (we don't want to train on the target variable directly)
if any(strcmp(heart_data.Properties.VariableNames, 'target'))
    heart_data(:, 'target') = [];
end

% Convert the table to a matrix (excluding the target column)
input_data = table2array(heart_data);

% Extract the target labels from the new data
% Assuming the 'cardio' column in heart_data is the actual target labels
new_target = heart_data.cardio;

% Split the data for training and testing (you can adjust the ratio as needed)
Icross_new = crossvalind('kfold', size(heart_data, 1), 8); % New fold indices
Itest_new = Icross_new == 1;
Itrain_new = ~Itest_new;

Xtrain_new = input_data(Itrain_new, :); % New training data (excluding target column)
ttrain_new = new_target(Itrain_new);    % New training targets

Xtest_new = input_data(Itest_new, :);   % New testing data
ttest_new = new_target(Itest_new);      % New testing targets

% Fine-tune the existing network with the new data
% Use a small learning rate to prevent catastrophic forgetting
net.trainParam.showWindow = 0; % Hide training window

% Re-training with new data, keeping the previous learned parameters
net = train(net, Xtrain_new', ttrain_new'); % Transpose data for fitting

% Predict the values for the new test data
ytest_new = sim(net, Xtest_new')';   % Predicted values for new test data

% Plot the error histogram for the updated model
ploterrhist(ytest_new - ttest_new, 'test');

% Save the updated model
save('UpdatedHeartDiseaseModel.mat', 'net');


%% cardio_train pred model

% Assuming Cardio_data is already loaded and is a table.
% The "cardio" column is the target (1 = has disease, 0 = no disease).

% Separate features (X) and target (y)
X = Cardio_data{:, 1:end-1};  % All columns except the last (features) 
y = Cardio_data{:, end};      % The last column (target: 0 or 1)

% Split data into training and testing sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% training, 20% testing

Xtrain = X(training(cv), :);
ytrain = y(training(cv), :);
Xtest = X(test(cv), :);
ytest = y(test(cv), :);

% Create and train the neural network
net = fitnet(50); % 10 hidden units (can be tuned)
net.trainParam.showWindow = false; % Suppress GUI

% Set data division (training/validation)
net.divideParam.trainRatio = 0.9;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0; % Test data is separate

% Train the network
net = train(net, Xtrain', ytrain'); % Transpose for neural network input

% Test the network
% Predict on training and testing data
ytrain_pred = net(Xtrain')';
ytest_pred = net(Xtest')';

% Round predictions to 0 or 1 (binary classification)
ytrain_pred_rounded = round(ytrain_pred);
ytest_pred_rounded = round(ytest_pred);

%Evaluate performance
% Calculate accuracy
train_accuracy = sum(ytrain_pred_rounded == ytrain) / numel(ytrain) * 100;
test_accuracy = sum(ytest_pred_rounded == ytest) / numel(ytest) * 100;

fprintf('Training Accuracy: %.2f%%\n', train_accuracy);
fprintf('Testing Accuracy: %.2f%%\n', test_accuracy);

% Confusion matrix for test data
confusionchart(ytest, ytest_pred_rounded);
title('Confusion Matrix for Test Data');

% Plot error histogram
figure;
errors = ytest_pred - ytest;
ploterrhist(errors);
title('Error Histogram');
