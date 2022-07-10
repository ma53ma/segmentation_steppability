% Cylinder1 v1
% Reference object: mug
% Diameter: 9.2cm, radiud: 4.6cm, height: 11.28cm  

close; clear; clc;

for size_idx= 1:4

    scaleReduceFactor = 100;
    rin = (2.2 + size_idx * 0.8)/scaleReduceFactor; 
    rout = rin*1.15;
    
    % Outer part of cone
    r = linspace(0,rout, 2);
    th = linspace(0,2*pi) ;
    [R,T] = meshgrid(r,th) ;
    X_outer= R.*cos(T);
    X_outer = X_outer';
    Y_outer= R.*sin(T);
    Y_outer = Y_outer';
    Z_outer= 2*(rout - R);
    Z_outer = Z_outer';
    surf(X_outer, Y_outer, Z_outer)
    hold on

    % Inner part of cone
    r = linspace(0,rin, 2);
    th = linspace(0,2*pi);
    [R,T] = meshgrid(r,th);
    X_inner = R.*cos(T);
    X_inner = X_inner';
    Y_inner = R.*sin(T);
    Y_inner = Y_inner';
    Z_inner = 2*(rin - R);
    Z_inner = Z_inner';
    surf(X_inner, Y_inner, Z_inner)

    % Bottom part of cone
    hold on
    xin = rin * cos(th); 
    xout = rout * cos(th);
    yin = rin * sin(th); 
    yout = rout * sin(th);
    
    X_bottom = [xin;xout];
    Y_bottom = [yin;yout];
    Z_bottom = zeros(2, length(xout));
    bottom=surf(X_bottom, Y_bottom, Z_bottom);

    size(X_outer)
    size(X_inner)
    size(X_bottom)

    X = [X_outer X_inner X_bottom];
    Y = [Y_outer Y_inner Y_bottom];
    Z = [Z_outer Z_inner Z_bottom];
    
  
    % View
    grid on
    axis equal
    xlabel('x axis');ylabel('y axis');zlabel('z axis')
    
    formatSpec = '%s_%05d%s';
    A = sprintf(formatSpec, "Cone", size_idx, '.stl');
    
    surf2stl(A, X,Y,Z)

end