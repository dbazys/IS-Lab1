% Dominykas Bazys 20240837 PEPfm-25
% IS Lab 1 + Naive Bayes

clc;
clear all;

% Reading apple and pear images 
% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');
% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
% 5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

% Selecting features for training (same as before)
x1 = [hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2 = [metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
T = [1;1;1;-1;-1]; % 1 for apple, -1 for pear

%-------------------------------------------------------%
%% Naive Bayes Training

% Separate features by class
apple_idx = find(T == 1);
pear_idx = find(T == -1);

% Calculate mean and variance for each feature per class
mean_x1_apple = mean(x1(apple_idx));
var_x1_apple = var(x1(apple_idx));
mean_x2_apple = mean(x2(apple_idx));
var_x2_apple = var(x2(apple_idx));

mean_x1_pear = mean(x1(pear_idx));
var_x1_pear = var(x1(pear_idx));
mean_x2_pear = mean(x2(pear_idx));
var_x2_pear = var(x2(pear_idx));

% Prior probabilities
prior_apple = length(apple_idx) / length(T);
prior_pear = length(pear_idx) / length(T);

%-------------------------------------------------------%
%% Testing
x1test = [hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_P3 hsv_value_P4];
x2test = [metric_A4 metric_A5 metric_A6 metric_P3 metric_P4];
Ttest = [1;1;1;-1;-1];

for i = 1:5
    % Likelihoods using Gaussian PDF
    p_x1_apple = normpdf(x1test(i), mean_x1_apple, sqrt(var_x1_apple));
    p_x2_apple = normpdf(x2test(i), mean_x2_apple, sqrt(var_x2_apple));
    prob_apple = p_x1_apple * p_x2_apple * prior_apple;

    p_x1_pear = normpdf(x1test(i), mean_x1_pear, sqrt(var_x1_pear));
    p_x2_pear = normpdf(x2test(i), mean_x2_pear, sqrt(var_x2_pear));
    prob_pear = p_x1_pear * p_x2_pear * prior_pear;

    % Classification decision
    if prob_apple > prob_pear
        ytest(i) = 1;
    else
        ytest(i) = -1;
    end

    etest(i) = Ttest(i) - ytest(i);
end

test_error = sum(abs(etest));
fprintf('Final error (Naive Bayes): %d\n', test_error);
