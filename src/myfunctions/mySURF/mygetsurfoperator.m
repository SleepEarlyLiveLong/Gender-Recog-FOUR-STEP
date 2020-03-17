function surfoperator = mygetsurfoperator(img)

[~,~,c] = size(img);
if c==3
    img = rgb2gray(img);
end

MetricThreshold = 1000;
NumOctaves = 3;
NumScaleLevels = 4;
% detect image feature points
points = detectSURFFeatures(img,'MetricThreshold',MetricThreshold,...
    'NumOctaves',NumOctaves,'NumScaleLevels',NumScaleLevels);  
% Extract feature descriptor
surfoperator = extractFeatures(img, points);
end