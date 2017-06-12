function mesg=dispstructure(structvar)
% This function outputs a caracter array with nested structures. May be enhanced....
% Author : B. Charlier (2017)


mesg = [];
structvar = orderfields(structvar);

    
    fields=fieldnames(structvar);
    for k=1:length(fields)
        if isstruct(eval(['structvar.' fields{k}]))
            str = orderfields(eval(['structvar.',fields{k}])); 
            mesg = [mesg,['        ',fields{k},'.',char(10)],evalc(' disp( str )')];  
            structvar = rmfield(structvar, fields{k});    
        end
    end


  mesg_root = evalc('disp(structvar)');
 
 mesg = [mesg_root,mesg];


end
