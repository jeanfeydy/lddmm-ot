function export_matching_tan(template,momentums,funres,data,summary,saveDir,format)
% EXPORT_MATCHING_TAN(template,momentums,funres,data,defo,saveDir,format) exports a 
% matching.
%
% EXPORT_MATCHING_TAN(template,momentums,funres,data,summary,saveDir,format)
% export a matching and create a summary file from the informations
% contained in the structure "summary". If the flag optim.save_template_evolution 
% was set to 1, it also exports the evolution of the mean template during the gradient
% descent.
%
% Note: If the directory saveDir already exists,  it is OVERWRITTEN without any warning.
%
% Input :
%   template: structure with template
%   momentums: cell of momentums
%   funres: cell of functional
%   data: structure with target
%   defo: structure containing deformation options
%   summary: structure given by the fsmatch_tan or jnfmean_tan_free function
%   saveDir (optional): directory to save the files  , use [] to set default : './LastMatching')
%   format (optional): string : 'ply' (Standford polygon Format, default) or 'vtk' (Visualization toolkit)
%
% See also : export_fshape_vtk, export_fshape_ply, export_atlas_HT, export_atlas_free, jnfmean_tan_free
% Author : B. Charlier (2017)

%----------------
%   Init
%----------------

if nargin==5
   saveDir = './LastMatching';
   format = 'vtk';
elseif nargin==6
   format = 'vtk';
end

switch format
case 'vtk'
      export_type = @export_fshape_vtk;
case 'ply'
      export_type = @export_fshape_ply;
end

if isempty(saveDir)
    saveDir = './LastMatching';
end
saveDir = saveDir(1:end-(strcmp('/',saveDir(end))));

try
    rmdir(saveDir,'s')
end
mkdir(saveDir)

if ~iscell(template)
    template={template};
end

if ~iscell(data)
    data= {data};
end

nb_match = size(template,2);

if ~( size(template,2)==size(data,2) )  
   error('template and data should have the same number of fshapes...') 
end

if iscell(momentums)
    momentums =cell2mat(momentums);
end

if isempty(funres)
    funres = cell(1,nb_match);
    for l=1:nb_shape
        [funres{:,l}]= deal(zeros(size(template{l}.x,1),1));
    end
elseif ~iscell(funres)
    funres = {funres};
end


%-------------
% Shooting plot
%-------------


templatextotal = cell2mat(cellfun(@(y) y.x,template,'uniformOutput',0)');
deflag = [0,cumsum(cellfun(@(y) size(y.x,1),template))];

% load deformation settings: defo or summary structure 
if isfield(summary,'parameters')
    defo = summary.parameters.defo;
    objfun = summary.parameters.objfun;
else
    defo = summary;
    objfun=[];
end

[Xt,~] = shoot_and_flow_tan(templatextotal,momentums,defo);

% save shootings

for i=1:length(Xt)
    for l=1:nb_match
        
        fname = [saveDir,'/',num2str(l),'-shoot','-',num2str(i),'.',format];
        if isempty(objfun) || strcmpi(objfun{l}.signal_type,'vertex')
            export_type(struct('x',Xt{i}(deflag(l)+1:deflag(l+1),:),'f',template{l}.f+funres{l}*(i-1)/defo.nb_euler_steps,'G',template{l}.G),fname)
        elseif strcmpi(objfun{l}.signal_type,'face')
            export_fshape_vtk(struct('x',Xt{i}(deflag(l)+1:deflag(l+1),:),'f',template{l}.f+funres{l}*(i-1)/defo.nb_euler_steps,'G',template{l}.G),fname,[],'face')
        end
    end
end

fprintf('\n')


 % save target
for l=1:nb_match
    fname = [saveDir,'/',num2str(l),'-target','.',format];
    if isempty(objfun) || strcmpi(objfun{l}.data_signal_type,'vertex')
        export_type(struct('x',data{l}.x,'f',data{l}.f,'G',data{l}.G),fname)
    elseif strcmpi(objfun{l}.data_signal_type,'face')
       export_type(struct('x',data{l}.x,'f',data{l}.f,'G',data{l}.G),fname,[],'face')
    end
end
    
% save funres
for l=1:nb_match
    fname = [saveDir,'/',num2str(l),'-funres.vtk'];
    if isempty(objfun) || strcmpi(objfun{l}.signal_type,'vertex')
        export_type(struct('x',Xt{1}(deflag(l)+1:deflag(l+1),:),'f',funres{l},'G',template{l}.G),fname);
    elseif strcmpi(objfun{l}.signal_type,'face')
        export_fshape_vtk(struct('x',Xt{1}(deflag(l)+1:deflag(l+1),:),'f',funres{l},'G',template{l}.G),fname,[],'face');
    end
end

% save momentum
fname = [saveDir,'/initial_momentum.vtk'];
export_mom_vtk(templatextotal,momentums,fname,'ascii')


    fprintf('\n')
 
% generate summary
if isfield(summary,'parameters')
    generate_summary(template,data,summary,saveDir)
end  


%saved evolution
if isfield(summary,'bfgs') &&  isfield(summary.bfgs,'xrec')

mkdir([saveDir,'/xrec']);
	for i = 1:size(summary.bfgs.xrec,2)

		fname = [saveDir,'/xrec/state_it',num2str(i),'.vtk'];
		momc = reshape(summary.bfgs.xrec(:,i),[],3);
		[Xt,~] = shoot_and_flow_tan(templatextotal,momc,defo);
		if isempty(objfun) || strcmpi(objfun{l}.signal_type,'vertex')
			export_type(struct('x',Xt{end}(deflag(l)+1:deflag(l+1),:),'f',funres{l},'G',template{l}.G),fname);
    		elseif strcmpi(objfun{l}.signal_type,'face')
		    	export_fshape_vtk(struct('x',Xt{end}(deflag(l)+1:deflag(l+1),:),'f',funres{l},'G',template{l}.G),fname,[],'face');
    		end

	end
end


end
