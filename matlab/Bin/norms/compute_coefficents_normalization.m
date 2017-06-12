function nobjfun=compute_coefficents_normalization(objfun,templateInit,data)
% nobjfun=COMPUTE_COEFFICENTS_NORMALIZATION(objfun,templateInit,data) returns the normalization coefficient that make the code scale invariant
% Author : B. Charlier (2017)

nb_match = size(data,2);

% Computing bounding box/normalization parameters
R    = sqrt(max([ max(cellfun(@(y) sum(var(y.x)),data)),max(cellfun(@(y) sum(var(y.x)),templateInit))]));

a = cell2mat(cellfun(@(y) y.f,data(:),'UniformOutput',0));a=a(:);
b = cell2mat(cellfun(@(y) y.f,templateInit(:),'UniformOutput',0));b=b(:);
Rf   = std([a;b]);
if Rf == 0
	Rf =1; % prevent division by zero
end

nobjfun =objfun;

for l = 1:nb_match

    if objfun{l}.normalize_objfun == 1
    
        nobjfun{l}.R  = R;
        nobjfun{l}.gC = R^(-2*( size(data{1,l}.G,2)-1)); % normalization for the term g in the energy
        nobjfun{l}.mC = R^(-2); % normalization for the deformation cost in the energy
        
        
        nobjfun{l}.dgxC= nobjfun{l}.gC*((R)^2); % normalization for the gradient_x of g
        nobjfun{l}.dgfC= nobjfun{l}.gC*((Rf)^2);% normalization for the gradient_f of g

    else
	R =1;
	Rf=1;
        nobjfun{l}.R  = R;
        nobjfun{l}.gC = R^(-2*( size(data{1,l}.G,2)-1)); % normalization for the term g in the energy
        nobjfun{l}.mC = R^(-2); % normalization for the deformation cost in the energy
        
        nobjfun{l}.dgxC= 1; % normalization for the gradient_x of g
        nobjfun{l}.dgfC= 1;% normalization for the gradient_f of g

    end
end

end
