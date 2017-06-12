function disp_iteration_info(nb_iter_total,ENRnew,stopCond,stepsNew,list_of_variables)
% Author : B. Charlier (2017)

x=['steps sizes :'];
for i = 1 : length(list_of_variables)
	x = [x ,' ',list_of_variables{i},' : %4.2e,'];
end

fprintf(['\nit. %3d : functional value : %4.2e, Stopping condition (MeandENR) : %4.2e \n          ',x(1:end-1),'\n\n'],...
		nb_iter_total,ENRnew,stopCond(1),stepsNew{1},stepsNew{2}, stepsNew{3},stepsNew{4});

end
