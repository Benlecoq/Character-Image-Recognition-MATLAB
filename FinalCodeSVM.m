%% Train SVM for Character Recognition
%-------------------------------------------------------------------------------------------
%-------------------------------------------------------------------------------------------
% WRITTEN BY - Benjamin Auzanneau
% DATE OF SUBMISSION - April/2017
% MATLAB VERSION - R2016b
%-------------------------------------------------------------------------------------------
% INDEX
% 1. Load dataset
% 2. Preproces of data
% 3. Change Data format
% 4. Extract HOG Features
% 5. Train default SVM and perform grid search
% 6. Graph the grid search
% 7. Find best parameter combination
% 8. Train final svm
% 9. Training Accuracy
% 10. Test final Model
% 11. Confusion Matrix
% 12. Train model on 10% of dataset and make predictions

%% 1. Load dataset

%imdb = load('/Users/benjaminauzanneau/Desktop/nncw/practical-cnn-2016a/data/charsdb.mat') ;
random = randperm(size(imdb.images.data,3));      % randomize order of data set

%% 2. Preproces of data

clf                 % Plot images before preprocessing
for i = 1:20
subplot(4,5,i);
imshow(data(:,:,i*1000));
end

Bi_data = zeros(size(data,1),size(data,2),size(data,3));    % binarize images
for i=1:size(data,3)
    Bi_data(:,:,i) = single(data(:,:,i)>0.8);               % binary threshold = 0.8
end

figure; clf         % Plot images after preprocessing
for i = 1:20
subplot(4,5,i);
imshow(Bi_data(:,:,i*1000));
end

%% 3. Change Data Format

Cell_data = cell(1,size(Bi_data,3));    % data in cell array format
for i=1:size(Bi_data,3)
    Cell_data{i} = Bi_data(:,:,i);
end

Column_data = zeros(size(Bi_data,1)*size(Bi_data,2),size(Bi_data,3)); % images in column format
for i=1:size(Bi_data,3)
    Column_data(:,i) = (reshape(Bi_data(:,:,i), 1, []))';
end

labels_dummy = (dummyvar(labels))';  % labels transform into dummy variables

test_ratio = .2;     % test set ratio
test_idx = randi([1 size(labels,2)],1,round(size(labels,2)*test_ratio));    % index of test data

Xtrain = Cell_data;                 % Cell array binary training data
Xtrain(:,test_idx) = [];     
Xtest = Cell_data(:,test_idx);        % Cell array binary test data

Ytrain = labels; 
Ytrain(:,test_idx) = [];            % train labels (numerical)    
Ytest = labels(:,test_idx);         % test labels (numerical)

%% 4. HOG Feature extraction

% HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(Cell_data{1,1},'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(Cell_data{1,1},'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(Cell_data{1,1},'CellSize',[8 8]);

% Show the original image
figure;
subplot(2,3,1:3); imshow(Cell_data{1,1});

% Visualize the HOG features
subplot(2,3,4);
plot(vis2x2);
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});

subplot(2,3,5);
plot(vis4x4);
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

subplot(2,3,6);
plot(vis8x8);
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});


% Extract HOG features data
HOG_data = zeros(size(Cell_data,2),324);
for i=1:size(Cell_data,2)
 HOG_data(i,:) = extractHOGFeatures(Cell_data{1,i},'CellSize', [8 8]); %change last part accordingly [4 4] [2 2]
end

HOGtrain = HOG_data;            % training set of HOG features
HOGtrain(test_idx,:) = [];

HOGtest = HOG_data(test_idx,:); % test set of HOG features

%% 5. Train default SVM and perform grid search

rng(1); % for reproducibility

%set values to be tested during grid search
kernelscale = [0.25 0.5 0.75 1 1.25]; 
boxconstraint = [0.25 0.5 0.75 1 1.25];

tic;

for i=1:5; 
    for j=1:5;
    
    t = templateSVM('KernelFunction','gaussian','KernelScale',kernelscale(i),'BoxConstraint',boxconstraint(j)) ; % SVM template

    SVMnet = fitcecoc(HOGtrain,Ytrain','Learners',t) ;                        

    CrosVal = crossval(SVMnet,'KFold',3); % 3 Fold Cross Validation 

    gridve(i,j) = mean(kfoldLoss(CrosVal,'Mode','individual')); % individual classification error 
    
    
    
    end;
end;

toc;
%% 6. Graph the grid search

X = [kernelscale];
Y = [boxconstraint];
subplot(1,1,1);
imagesc(gridve), axis equal tight, colorbar;
set(gca,'Xdir','Normal','XTick', 1:5, 'XTickLabel', kernelscale);
set(gca, 'Ydir','Normal','YTick', 1:5, 'YTickLabel',boxconstraint);
title('Hyperparameter Grid Search');
xlabel('Kernel Scale');
ylabel('Box Constraint');
colormap pink;


%% 7. Find best parameter combination

[optimalkernelscale,optimalboxconstraint] = find(gridve ==min(min(gridve))) %find smallest error value
optimalvalidationerror = min(min(gridve)) %return corresponding parameter combination

%% 8. Train final svm

tic;

rng(1); % for reproducibility

tfinal = templateSVM('KernelFunction','gaussian','KernelScale',1,'BoxConstraint',1.25) ; % SVM template

SVMnetfinal = fitcecoc(HOGtrain,Ytrain','Learners',tfinal) ;             

CrosVal = crossval(SVMnetfinal,'KFold',3); % 3 Fold Cross Validation 

finalerror = mean(kfoldLoss(CrosVal,'Mode','individual')); % individual classification loss 

toc;


%% 9. Training Accuracy

train = predict(SVMnetfinal,HOGtrain);
train_acu = sum(Ytrain'==train)/size(Ytrain,2);


%% 10. Test final Model

Yhat = predict(SVMnetfinal,HOGtest);
test_acu = sum(Ytest'==Yhat)/size(Ytest,2);

%% 11. Confusion Matrix

% Display the confusion matrix in a formatted table.
confuMat = confusionmat(Ytest',Yhat);
confuMat = bsxfun(@rdivide,confuMat,sum(confuMat,2)); % Convert confusion matrix into percentage form

digits = 'a':'z';
colHeadings = arrayfun(@(x)sprintf('%s',x),'a':'z','UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'letters |   ',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '       |  ']);
    fprintf('%-9.2f', confuMat(idx,:));
    fprintf('\n')
end

% export confusion matrix as table
confuMat_csv = array2table(confuMat);
confuMat_csv.Properties.VariableNames = cellstr(digits');
confuMat_csv.Properties.RowNames = cellstr(digits');
writetable(confuMat_csv,'confuMat.csv')

% display example missclassified characters
wrong_x     = Xtest(CYtest~=CYhat_test);
wrong_y     = CYtest(CYtest~=CYhat_test);
wrong_yhat  = CYhat_test(CYtest~=CYhat_test);
figure; clf
j = 70;
for i = 1:9
    subplot(3,3,i);
    imshow(wrong_x{1,j});
    title({['Predicted = ', digits(wrong_yhat(1,j))]; ['True = ', digits(wrong_y(1,j))]});
    j=j+1;
end

%% 12. Train model on 10% of dataset and use it to predict the remaining 90%

%Perform same operations as previously but make test set ratio 0.1. The
%test set will actually be used to train and the training set to test

%Change Data Format

Cell_data = cell(1,size(Bi_data,3));    % data in cell array format
for i=1:size(Bi_data,3)
    Cell_data{i} = Bi_data(:,:,i);
end

Column_data = zeros(size(Bi_data,1)*size(Bi_data,2),size(Bi_data,3)); % images in column format
for i=1:size(Bi_data,3)
    Column_data(:,i) = (reshape(Bi_data(:,:,i), 1, []))';
end

labels_dummy = (dummyvar(labels))';  % labels transform into dummy variables

test_ratio = .1;     % ALTERED test set ratio
test_idx = randi([1 size(labels,2)],1,round(size(labels,2)*test_ratio));    % index of test data

Xtrain = Cell_data;                 % Cell array binary training data
Xtrain(:,test_idx) = [];     
Xtest = Cell_data(:,test_idx);        % Cell array binary test data

Ytrain = labels; 
Ytrain(:,test_idx) = [];            % train labels (numerical)    
Ytest = labels(:,test_idx);         % test labels (numerical)


% HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(Cell_data{1,1},'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(Cell_data{1,1},'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(Cell_data{1,1},'CellSize',[8 8]);

% Show the original image
figure;
subplot(2,3,1:3); imshow(Cell_data{1,1});

% Visualize the HOG features
subplot(2,3,4);
plot(vis2x2);
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});

subplot(2,3,5);
plot(vis4x4);
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

subplot(2,3,6);
plot(vis8x8);
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});


% Extract HOG features data
HOG_data = zeros(size(Cell_data,2),324);
for i=1:size(Cell_data,2)
 HOG_data(i,:) = extractHOGFeatures(Cell_data{1,i},'CellSize', [8 8]); %change last part accordingly [4 4] [2 2]
end

HOGtrain = HOG_data;            % training set of HOG features
HOGtrain(test_idx,:) = [];

HOGtest = HOG_data(test_idx,:); % test set of HOG features


% train model on 10%

rng(1); % for reproducibility

t01 = templateSVM('KernelFunction','gaussian','KernelScale',1,'BoxConstraint',1.25) ; % SVM template

SVMnet01 = fitcecoc(HOGtest,Ytest','Learners',t01) ;   %Use test set as it represents 10% of data         

CrosVal01 = crossval(SVMnet01,'KFold',3); % 3 Fold Cross Validation 

finalerror01 = mean(kfoldLoss(CrosVal01,'Mode','individual')); % individual classification loss 


% Make predictions with model trained on 10%

Yhat10 = predict(SVMnet01,HOGtrain);
test_acu10 = sum(Ytrain'==Yhat10)/size(Ytrain,2);

