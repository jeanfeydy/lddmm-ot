function [nobjfun,mesg] = set_objfun_option(objfun,templateInit,data,list_of_variables)
% This function check the objfun structure and set default values if needed.
% Author : B. Charlier (2017)

if ~iscell(objfun)
    objfun ={objfun};
    objfun = objfun(ones(1,size(data,2)));
end

nb_match = size(data,2);
if ~isequal(size(templateInit,2),length(objfun),nb_match)
    error('The initial template, objfun and data must be of length nb_match')
end

nobjfun = objfun;

is_HT_algo = (sum(strcmp(list_of_variables(:),'pHT')) >0);
is_matching_algo = (sum(strcmp(list_of_variables(:),'p')) >0) && (length(list_of_variables) == 1) ;

for l = 1:nb_match
    
    nobjfun{l} = setoptions(nobjfun{l}, 'distance', 'kernel',{'kernel','wasserstein','pt2pt'});
    
    switch lower(nobjfun{l}.distance)
        case 'kernel'
            nobjfun{l}.kernel_distance = set_kernel_distance_defaults(nobjfun{l}.kernel_distance);
            
        case 'wasserstein'
            nobjfun{l}.wasserstein_distance = set_wasserstein_distance_defaults(nobjfun{l}.wasserstein_distance);
            
        case 'pt2pt'
            if (isequal(cell2mat(cellfun(@(y) y.x,data)), ( repmat( cellfun(@(y) y.x,templateInit),size(data,1) ,1)) ))
                error('Check data and template sizes to use pt2pt distance');
            end
    end
    

   
    nobjfun{l} = setoptions(nobjfun{l}, 'weight_coef_dist', 40);
    nobjfun{l} = setoptions(nobjfun{l}, 'weight_coef_pen_p', 1);
    
    nobjfun{l} =  set_femType_defaults(nobjfun{l},templateInit{l},data{l});
    
    
    if ~is_matching_algo
        nobjfun{l} = setoptions(nobjfun{l}, 'weight_coef_pen_f', .01);
        nobjfun{l} = setoptions(nobjfun{l}, 'weight_coef_pen_fr', .01);
        
        nobjfun{l} = set_functional_distance_defaults(nobjfun{l});
    end
    
    if is_HT_algo
        nobjfun{l} = setoptions(nobjfun{l}, 'weight_coef_pen_pHT', .01);
    end
    
    nobjfun{l} = setoptions(nobjfun{l}, 'normalize_objfun', 0);
    
    
    
    
end

% check that 'weight_coef_pen_*'  are the same for the momentums :
if ~isequal(cellfun(@(x) x.weight_coef_pen_p,nobjfun), ones(1,nb_match)*nobjfun{1}.weight_coef_pen_p)
    error('weight_coef_pen_p should be the same for all the objfun structure')
end
if is_HT_algo && ~isequal(cellfun(@(x) x.weight_coef_pen_pHT,nobjfun), ones(1,nb_match)*nobjfun{1}.weight_coef_pen_pHT)
    error('weight_coef_pen_pHT should be the same for all the objfun structure')
end


% save message and display
mesg = sprintf('\n----------- Objective function parameters -----------\n' );
for l=1:nb_match
    mesg = [mesg,dispstructure(nobjfun{l})];
end
fprintf('%s',mesg);
end

function obj = set_wasserstein_distance_defaults(obj)

obj = setoptions(obj,'epsilon');
obj = setoptions(obj,'niter',1000);
obj = setoptions(obj,'tau', 0); % basic sinkhorn, no extrapolation
obj = setoptions(obj,'rho',Inf); % balanced case
obj = setoptions(obj,'weight_cost_varifold',[1,.01]); % weight on spatial and orientation distances
obj = setoptions(obj, 'method', 'matlab',{'cuda','matlab'});

end

function obj = set_kernel_distance_defaults(obj)

obj = setoptions(obj,'distance','empty');

if ~strcmp(obj.distance,'empty')
    switch lower(obj.distance)
        case 'cur'
            obj.kernel_geom = 'gaussian';
            obj.kernel_signal = 'gaussian';
            obj.kernel_grass = 'linear';
        case 'var'
            obj.kernel_geom = 'gaussian';
            obj.kernel_signal = 'gaussian';
            obj.kernel_grass = 'binet';
        case 'varexpo'
            obj.kernel_geom = 'gaussian';
            obj.kernel_signal = 'gaussian';
            obj.kernel_grass = 'gaussian_unoriented';
        otherwise
            obj.distance = 'var';
            obj.kernel_geom = 'gaussian';
            obj.kernel_signal = 'gaussian';
            obj.kernel_grass = 'binet';
            warning('distance : Possible distance are current (''cur'') or varifold (''var'' or ''varexpo''). objfun.distance is set to ''var''.')
    end
       
elseif ~isfield(obj,'kernel_geom') ||~isfield(obj,'kernel_signal') || ~isfield(obj,'kernel_grass')
    error('Please provide a field distance (cur, var or varexpo) or kernel type for geometry, signal and grassmanian');
end

obj = setoptions(obj,'kernel_size_geom'); % size of the geometric kernel  in the data attachment term
obj = setoptions(obj,'kernel_size_signal');% size of the functional kernel in the data attachment term
if ~strcmpi(obj.kernel_grass,'binet') && ~strcmpi(obj.kernel_grass,'linear')
    obj = setoptions(obj,'kernel_size_grass');
end

obj = setoptions(obj, 'method', 'matlab',{'cuda','matlab'});

end


function obj = set_functional_distance_defaults(obj)


obj = setoptions(obj,'pen_signal','l2');

switch lower(obj.pen_signal)
    case 'bv'
        
        obj = setoptions(obj,'weight_coef_pen_signal',1);
        obj = setoptions(obj,'weight_coef_pen_dsignal',1);obj.fem_type
        obj = setoptions(obj,'fem_type','p2',{'p2'});
        
        if  ~isfield(obj,'norm_eps')
            obj.norm_eps=1e-6;
        end
        
    case 'h1'
        
        obj = setoptions(obj,'weight_coef_pen_signal',1);
        obj = setoptions(obj,'weight_coef_pen_dsignal',1);
        obj = setoptions(obj,'fem_type','p2',{'p2'});
end

end

function obj = set_femType_defaults(obj,templateInit,data)


% Try to detect which fem_type to use fot the template
    tempP = size(templateInit.x,1);
    tempM = size(templateInit.G,1);
    tempF = size(templateInit.f,1);
    
    if (tempP ~= tempM)
        
        if (tempP == tempF)
            obj = setoptions(obj,'signal_type','vertex',{'face','vertex'});
        elseif (tempM == tempF)
            obj = setoptions(obj,'signal_type','face',{'face','vertex'});         
        else
            error('Dimension mismatch for signal template')
        end
        
    elseif (tempP == tempM)
        
        obj = setoptions(obj,'signal_type');
        
    end
    
    if (strcmpi(obj.signal_type,'face') && (tempM ~= tempF)) ||...
            ( strcmpi(obj.signal_type, 'vertex') && (tempP ~= tempF))
        error('Dimension mismatch for signal')
    end    
    
    if strcmpi(obj.signal_type,'face')
        obj = setoptions(obj,'fem_type','p1',{'p1'});
    
    elseif strcmpi(obj.signal_type,'vertex')
        obj = setoptions(obj,'fem_type','p2',{'p2','lump'});
    
    end
      
    

% Try to detect which fem_type to use fot the data    
    dataM = size(data.G,1);
    dataP = size(data.x,1);
    dataF = size(data.f,1);
    
    if (dataP ~= dataM)
        
        if (dataP == dataF)
            obj = setoptions(obj,'data_signal_type','vertex',{'face','vertex'});
        elseif (dataM == dataF)
            obj = setoptions(obj,'data_signal_type','face',{'face','vertex'});
        else
            error('Dimension mismatch for signal data')
        end
        
    elseif (dataP == dataM) 
         obj = setoptions(obj,'data_signal_type');
    
    end
    
    if (strcmpi(obj.data_signal_type,'face') && (dataM ~= dataF)) ||...
            ( strcmpi(obj.data_signal_type, 'vertex') && (dataP ~= dataF))
        error('Dimension mismatch for signal data')
    end
     

end

