load('labels');

% vgg 0.25
train_fea=load('trainImagePool5Feature25');
test_fea=load('testImagePool5Feature25');
model = svmtrain(labels_train,train_fea.pool5fea);
[~, accuracy, dec_values] = svmpredict(labels_test, test_fea.pool5fea, model); % test the training data

% vgg 0.5
train_fea=load('trainImagePool5Feature5');
test_fea=load('testImagePool5Feature5');
model = svmtrain(labels_train, train_fea.pool5fea);
[predict_label, accuracy, dec_values] = svmpredict(labels_test, test_fea.pool5fea, model); % test the training data
