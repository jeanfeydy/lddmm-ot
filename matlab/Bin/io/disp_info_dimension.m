function msg = disp_info_dimension(data,template,silent)
% print some informations about the dimension of the problem
% Author : B. Charlier (2017)

deflag = [0,cumsum(cellfun(@(y) size(y.x,1),template))]; 
nlist = diff(deflag); % number of point in each shape

[nb_obs,nb_match] = size(data);
x = repmat(' %d,',1,nb_match);
msg = [sprintf('\n----------- Dimensions of the problem -----------\n'),...
sprintf('Data : %d observations containing %d fshape(s) each.',nb_obs,nb_match),...
sprintf([' The mean number of points in each shape is : ',x(1+1:end-1),'.'],floor(mean(cellfun(@(y) size(y.x,1),data)))),...
sprintf('\nMean Template : contains %d fshape(s). The number of points is : ',size(template,2)),...
sprintf([x(1+1:end-1),'.'],nlist),...
sprintf('\n')];

if nargin ==2 || (silent == 0)
    disp(msg)
end

end
