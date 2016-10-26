% 2. Cordinate turn model
function f2 = c_turn(x,param)
    dt = param;
    if x(5) == 0
        f2 =     [x(1) + x(3) * dt;
                  x(2) + x(4) * dt;
                  x(3);
                  x(4);
                  0              ];
    else
        wt = x(5) * dt;
        w = x(5);
        f2 =     [x(1) + x(3)/w * sin(wt) - x(4)/w * (1 - cos(wt));
                  x(2) + x(3)/w* (1 - cos(wt)) + x(4)/w*sin(wt);
                  x(3)*cos(wt) - x(4) * sin(wt);
                  x(3) * sin(wt) + x(4)* cos(wt);
                  w;                                           ];
    end

end 