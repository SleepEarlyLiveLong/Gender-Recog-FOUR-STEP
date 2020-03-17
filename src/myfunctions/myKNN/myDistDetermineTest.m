function predicted_label = myDistDetermineTest(core,testvector)

if ~ismatrix(core)||~isvector(testvector)
    error('Input parameters Error!');
end
testvector = testvector(:)';
if size(core,2)~=size(testvector,2)
    error('Input parameters Error!');
end
core_num = size(core,1);
dist = zeros(core_num,1);
for i=1:core_num
    dist(i) = sum((core(i,:)-testvector).^2);
end
[~,pos] = min(dist);
predicted_label = pos;
end