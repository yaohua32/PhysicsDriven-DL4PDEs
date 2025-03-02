%%%%%%%%%%%%%%%%%%
%PDE格式为：
% m*u_tt  - \nabla(c \nabla u) + a * u = f
% 需要设定参数：m, d=0, c, a, f
%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%
clear; clc;
tmax = 5.; nt=100;
%
model = createpde();
%%%%%%%%%%%%%%%%%% Define the domain: [-1,1]^2
R = [3,4,0,1,1,0,1,1,0,0]';
g = decsg(R);
geometryFromEdges(model, g);
% pdegplot(model, "EdgeLabels","on","FaceLabels","on")
 %%%%%%%%%%%%%%%%%%
specifyCoefficients(model,"m", 1, "d", 0, "c", 1, "a", 0, "f", 0.);
%%%%%%%%%%%%%%%%%%%% The FEM mesh
generateMesh(model, "Hmax", 0.1);
% figure(1); pdemesh(model);
%%%%%%%%%%%%%%%%%% The boundary condition (zero boundary condition)
applyBoundaryCondition(model,"dirichlet","Edge",[1, 2, 3, 4],"u",0);
%%%%%%%%%%%%%%%%%% The initial condition
u0 = @(location) 4. .* (location.x).^2 .* location.y .* ...
    (1-location.x) .* (1-location.y);
ut0 = @(location) 0. .* location.x + 0. * location.y;
setInitialConditions(model, u0, ut0);
%%%%%%%%%%%%%%%%%%%% solve the pde
tlist = linspace(0, tmax, nt);
% model.SolverOptions.ReportStatistics ='on';
results = solvepde(model, tlist);
xgrid = results.Mesh.Nodes;
tgrid = results.SolutionTimes;
u = results.NodalSolution;
dux = results.XGradients;
duy = results.YGradients;
%%%%%%%%%%%%% Save the data
% save('./truth_2d.mat', 'u', 'xgrid', 'tgrid', '-v7.3')
%%%%%%%%%%%%%%
figure; subplot(2,2,1)
pdeplot(model,"XYData",u(:,1), "ZData",u(:,1), ...
    "ZStyle","continuous","Mesh","off")
title('u0');
subplot(2,2,2)
pdeplot(model,"XYData",u(:,end), "ZData",u(:,end), ...
    "ZStyle","continuous","Mesh","off")
title('uT');
subplot(2,2,3)
pdeplot(model,"XYData",dux(:,end), "ZData",dux(:,end), ...
    "ZStyle","continuous","Mesh","off")
title('duxT');
subplot(2,2,4)
pdeplot(model,"XYData",duy(:,end), "ZData",duy(:,end), ...
    "ZStyle","continuous","Mesh","off")
title('duyT');

