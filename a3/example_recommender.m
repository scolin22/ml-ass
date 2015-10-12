load newsgroups.mat

[N,D] = size(X);
K = 5;

model = recommender(X, K);

for n = 1:5
    wordNumbers = model.predict(model, n);
    fprintf('%s: ',wordlist{n});
    fprintf('%s ',wordlist{wordNumbers});
    fprintf('\n');
end
