function []=export_mom_vtk(pos,mom,fname,encod_type)
% Author : B. Charlier (2017)

if nargin ==3
    encod_type = 'ascii';
end

if size(pos,2)<=2
    pos = [pos,zeros(size(pos,1),3-size(pos,2))];
end

if size(mom,2)<=2
    mom = [mom,zeros(size(mom,1),3-size(mom,2))];
end


nb_points = size(mom,1);

fid = fopen(fname, 'w'); 


%-------------
%  header
%-------------

%ASCII file header
fprintf(fid, '# vtk DataFile Version 3.0\n');
fprintf(fid, 'VTK from fshapesTk\n');
if strcmp(encod_type,'ascii')
    fprintf(fid, 'ASCII\n\n');
else
    fprintf(fid, 'BINARY\n\n');
end

%-------------
%  Position
%-------------

%ASCII sub header
fprintf(fid, 'DATASET STRUCTURED_GRID\n');
fprintf(fid, ['DIMENSIONS ',num2str(nb_points),' 1 1\n']);
%Record position
fprintf(fid, ['POINTS ',num2str(nb_points),' float\n']); 
if strcmp(encod_type,'ascii')
    fprintf(fid,'%G %G %G\n',pos');
else
    fwrite(fid,pos','float','b');
end

%-------------
%  vectors
%-------------

%ASCII sub header
fprintf(fid, ['\nPOINT_DATA ',num2str(nb_points),'\n']);
%Record vectors
fprintf(fid, 'VECTORS momentum float\n');
if strcmp(encod_type,'ascii')
    fprintf(fid,'%G %G %G\n',mom');
else
    fwrite(fid, mom','float','b');
end

fclose(fid);

fprintf('\nFile saved in %s',fname)

end
