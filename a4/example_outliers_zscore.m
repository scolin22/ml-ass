load cities.mat

[n,d] = size(ratings);

z_scores = zscore(ratings,0,1);

[outCities,outRatings] = find(abs(z_scores) >= 4);

cities = names(outCities,:);
ratings = categories(outRatings,:);

for row = 1:size(cities,1)
    fprintf('City: %s, Category: %s\n', cities(row,:), ratings(row,:));
end
