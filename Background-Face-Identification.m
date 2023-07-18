%Assignment 2

%Housekeeping
clc;
close all;
warning('off', 'Images:initSize:adjustingMag');

% FILE DIRECTORY FACE TRAINING
folder = 'trainingData\\face';
ext = 'jpg'; %extension, no dot
% Get list of all files
content = dir(folder);
allfiles = {content.name};
isExt = endsWith(allfiles,['.',ext],'IgnoreCase',true);  
files = allfiles(isExt);
I = numel(files); %I is equal to the total number of face images

% DETERMINING MU FOR FACE IMAGES
meanFaces = 0;
for K = 1:I
    %Retrieving the data of the current image
    x_i = double(imread(fullfile(folder,files{K})));
    %ax = subplot(nrows,ncols, K);
    %imshow(this_image, 'Parent', ax);
    
    % xi is 1 image
    % make xi into a 1x900 vector
    data = x_i(:);
    meanFaces = meanFaces + data;
end
meanFaces = meanFaces / I;

% DETERMINING SIGMA FOR FACE IMAGES
sigmaFaces = 0;
for K = 1:I
    %Retrieving the data of the current image
    x_i = double(imread(fullfile(folder,files{K})));
    data = x_i(:);
    sub = data - meanFaces;
    product = sub * transpose(sub);
    sigmaFaces = sigmaFaces + product;
end
sigmaFaces = sigmaFaces / I;

% FILE DIRECTORY BACKGROUND TRAINING
folder = 'trainingData\\background';
ext = 'jpg'; %extension, no dot

% Get list of all files
content = dir(folder);
allfiles = {content.name};
isExt = endsWith(allfiles,['.',ext],'IgnoreCase',true);  
files = allfiles(isExt);
J = numel(files);

% DETERMINING MU FOR BACKGROUND IMAGES
meanBackgrounds = 0;
for K = 1:J
    %Retrieving the data of the current image
    x_i = double(imread(fullfile(folder,files{K})));
    data = x_i(:);
    meanBackgrounds = meanBackgrounds + data;
end
meanBackgrounds = meanBackgrounds / J;

% DETERMINING SIGMA FOR BACKGROUND IMAGES
sigmaBackgrounds = zeros(900, 900);
for K = 1:J
    %Retrieving the data of the current image
    x_i = double(imread(fullfile(folder,files{K}))); 
    data = x_i(:);
    sub = data - meanBackgrounds;
    product = sub * transpose(sub);
    sigmaBackgrounds = sigmaBackgrounds + product;
end
sigmaBackgrounds = sigmaBackgrounds / J;

% Run a for loop for both the face images and background images and
% calculate norm for both. If Norm(u1, E1) > Norm(u0, E0), then it is a 
% face photo. Otherwise it is a background photo.
% FILE DIRECTORY FACE TESTING
folder = 'testingData\\face';
ext = 'jpg'; %extension, no dot

% Get list of all files
content = dir(folder);
allfiles = {content.name};
isExt = endsWith(allfiles,['.',ext],'IgnoreCase',true);  
files = allfiles(isExt);
A = numel(files); % Total number of face testing images, can be adjusted

numFaces = 0; % Total number of face images correctly identified
numBackgrounds = 0; % Total number of background images correctly identified
for K = 1:A
    %Retrieving the data of the current image
    x_i = double(imread(fullfile(folder,files{K})));
    data = x_i(:);
    
    % SOLVING FOR PR(Y=1|X)
    logSum1 = 0;
    logSum0 = 0;
    vecSum1 = 0;
    vecSum0 = 0;
    for i = 1:900
        logSum1 = logSum1 + log(sigmaFaces(i,i));
        logSum0 = logSum0 + log(sigmaBackgrounds(i,i));
        vecSum1 = vecSum1 + ((data(i) - meanFaces(i))/(2 * sigmaFaces(i,i)));
        vecSum0 = vecSum0 + ((data(i) - meanBackgrounds(i))/(2 * sigmaBackgrounds(i,i)));
    end
    Pr_y_equalTo_1_given_x = (-0.5 * logSum1) - vecSum1;
    Pr_y_equalTo_0_given_x = (-0.5 * logSum0) - vecSum0;
    if(Pr_y_equalTo_1_given_x > Pr_y_equalTo_0_given_x)
        numFaces = numFaces + 1;
    end
end

% FILE DIRECTORY BACKGROUND TESTING
folder = 'testingData\\background';
ext = 'jpg'; %extension, no dot

% Get list of all files
content = dir(folder);
allfiles = {content.name};
isExt = endsWith(allfiles,['.',ext],'IgnoreCase',true);  
files = allfiles(isExt);
B = numel(files); % Total number of background testing images, can be adjusted

for K = 1:B
    %Retrieving the data of the current image
    x_i = double(imread(fullfile(folder,files{K})));
    data = x_i(:);
    
    % SOLVING FOR PR(Y=1|X) AND PR(Y=0|X)
    logSum1 = 0;
    logSum0 = 0;
    vecSum1 = 0;
    vecSum0 = 0;
    
    % USE BAYESIAN RULE TO TEST THE DATASET
    for i = 1:900
        logSum1 = logSum1 + log(sigmaFaces(i,i));
        logSum0 = logSum0 + log(sigmaBackgrounds(i,i));
        vecSum1 = vecSum1 + ((data(i) - meanFaces(i))^2/(2 * sigmaFaces(i,i)));
        vecSum0 = vecSum0 + ((data(i) - meanBackgrounds(i))^2/(2 * sigmaBackgrounds(i,i)));
    end
    % ASSUME PRIORS (PR(Y=0) and PR(Y=1)) ARE UNIFORM 
    Pr_y_equalTo_1_given_x = (-0.5 * logSum1) - vecSum1;
    Pr_y_equalTo_0_given_x = (-0.5 * logSum0) - vecSum0;
    if(Pr_y_equalTo_0_given_x > Pr_y_equalTo_1_given_x)
        numBackgrounds = numBackgrounds + 1;
    end
end

% CLASSIFICATION
fprintf('Face Image Accuracy: %f out of %f, %f %% \n', numFaces, A, (numFaces/A)*100);
fprintf('Background Image Accuracy: %f out of %f, %f %% \n', numBackgrounds, B, (numBackgrounds/B)*100);
