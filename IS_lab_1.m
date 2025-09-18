% Dominykas Bazys 20240837 PEPfm-25
% IS Lab 1

clc;
clear all;
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

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5

x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];

%Desired output vector
T=[1;1;1;-1;-1];

%-------------------------------------------------------%
%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);
%rng(1); % Seed
% calculate weighted sum with randomly generated parameters

for i = 1:5 % evaluation loop
    v(i) = x1(i)*w1 + x2(i)*w2 + b; % weighted sum
    if v(i) > 0
        y(i) = 1;
    else
        y(i) = -1;
    end
    e(i) = T(i) - y(i);
end

total_error = sum(abs(e)); %error

%-------------------------------------------------------%
%% Training

eta = 0.1; % Learning rate
epoch = 0; % To count epochs

e = 1; % just to enter the loop
while e ~= 0
    e = 0; % reset total error for this epoch
    epoch = epoch + 1; % increment epoch counter

    for i = 1:5
        % Compute weighted sum
        v = x1(i)*w1 + x2(i)*w2 + b;
       
        if v > 0
            y = 1;
        else
            y = -1;
        end
       
        e_i = T(i) - y;
        
        % Update weights and bias
        w1 = w1 + eta * e_i * x1(i);
        w2 = w2 + eta * e_i * x2(i);
        b  = b  + eta * e_i;
        
        % Accumulate total error
        
        e = e + abs(e_i);
        
        
    end
    fprintf('Current error: %d\n', e);
    % Display progress
    fprintf('Epoch: %d\n', epoch);
    %fprintf('Total error: %d\n', e);
    fprintf('Weights: w1=%.3f, w2=%.3f, b=%.3f\n\n', w1, w2, b);
end


%-------------------------------------------------------%
%% Testing

%A4,A5,A6,P3,P4
%building matrix 2x5
x1test=[hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_P3 hsv_value_P4];
x2test=[metric_A4 metric_A5 metric_A6 metric_P3 metric_P4];

Ttest=[1;1;1;-1;-1];

% calculate weighted sum with generated weight parameters

for i = 1:5
    vtest(i) = x1test(i)*w1 + x2test(i)*w2 + b;
    if vtest(i) > 0
        ytest(i) = 1;
    else
        ytest(i) = -1;
    end
    etest(i) = Ttest(i) - ytest(i);
end

test_error = sum(abs(etest));
fprintf('Final error: %d\n', test_error);