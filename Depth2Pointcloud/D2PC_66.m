clear;
angles = load('point66.txt');

cat = 'A_real_';
num = '000';
output_name = string('test/temp/')+string(cat)+string('/D/')+string(cat)+string('dep_num')+string(num)+string('_v');
file = output_name;
result = [];
%[0,1,2,3,4,5,12,13]
for iter = [0,1,2,3,4,5]
    img_name = file + num2str(iter,'%02d') + string('.jpg');
    img = imread(char(img_name));
    sum = 0;
    uvw = [];  
    for i = 1:256
        for j = 1:256
            if (img(i,j) < 140)
                temp = zeros(1,3);
                sum = sum + 1;
                temp(:,1) = j-128;
                temp(:,2) = 128-i;
                temp(:,3) = img(i,j);
                temp = double(temp)/128.0;  
                uvw(sum,:) = temp; %#ok<SAGROW>
            end
        end
    end
    
    % compute the transform matrix A.
    A = zeros(3,3);
    x0 = angles(iter+1,1);
    y0 = angles(iter+1,2);
    z0 = angles(iter+1,3);
    sign_x = sign(x0);
    sign_y = sign(y0);
    
    disp(x0);
    disp(y0);
    disp(z0);
    % if distance to 0  < 0.01, regard x0/y0 as 0
    if abs(x0*y0*z0) > 0.01 % x0 y0 z0 != 0
        %compute special A
        
        
        
    else  % on the line
        if abs(x0) > 0.01
            %disp('x!=0');
            cosa = z0; 
            sina = sign_x*sqrt(1-z0*z0);
            cosb = sign_x*x0/sqrt(1-z0*z0); 
            sinb = sign_x*y0/sqrt(1-z0*z0);
            A(1,1) = cosa*cosb; A(1,2) = -1*sinb; A(1,3) = -sina*cosb;
            A(2,1) = cosa*sinb; A(2,2) = cosb;    A(2,3) = -sina*sinb;
            A(3,1) = -1*sina;   A(3,2) = 0;       A(3,3) = -cosa;       
        else
            if z0 > -0.01  % z0 decides the direction of u&v.
                A(1,1) = 1;  A(1,2) = 0;     A(1,3) = 0;
                A(2,1) = 0;  A(2,2) = z0;    A(2,3) = -y0;
                A(3,1) = 0;  A(3,2) = -1*y0; A(3,3) = -z0;
            else
                A(1,1) = -1; A(1,2) = 0;     A(1,3) = -0;
                A(2,1) = 0;  A(2,2) = -z0;   A(2,3) = -y0;
                A(3,1) = 0;  A(3,2) = y0;    A(3,3) = -z0;
            end
        end
    end
    
    disp(A);

    
    %compute xyz: xyz = A*uvw + x0y0z0.
    xyz = A*uvw';  %3*3x3*n = 3*n
    [h,w] = size(xyz); 
    xyz0 = angles(iter+1,:);  % 1*3
    Bias = xyz0'*ones(1,w); % 3*1x1*n = 3*n
    xyz = (xyz + Bias)';  % n*3
    result = [result; xyz]; %#ok<AGROW>
    %[h1,w1] = size(xyz); 
    %fi = fopen(char(output_name+ num2str(iter-1,'%02d')+ string('.txt')), 'w');
    %for i = 1:h1
     %   fprintf(fi,'%.5f %.5f %.5f\n',xyz(i,:));
    %end
end
[h1,w1] = size(result); 
out_dir = string('test/temp/')+string(cat);
fi = fopen(char(string(out_dir)+ string('/Union_num_')+string(num)+ string('.txt')), 'w');
for i = 1:h1
    fprintf(fi,'%.5f %.5f %.5f\n',result(i,:));
end










