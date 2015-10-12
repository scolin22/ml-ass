load newsgroups.mat

[N,D] = size(X);
K = 4;

model = recommender(X, K);
rec = model.predict(model, 1);
