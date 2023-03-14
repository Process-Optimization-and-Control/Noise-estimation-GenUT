fileID = fopen("N_lhs.txt", "r");
formatSpec = '%f';
N_lhs = fscanf(fileID,formatSpec);
fclose(fileID);
d = load("par_posterior.mat");
par_post = d.par_posterior';
lhs_samples = lhs_empirco(par_post, N_lhs);
current_directory = "C:\Users\halvorak\Dokumenter\SUBPRO Digital twin\Noise-estimation-GenUT\scripts\lhs_iman_conover";
save(fullfile(current_directory, "lhs_samples_from_matlab.mat"), "lhs_samples")