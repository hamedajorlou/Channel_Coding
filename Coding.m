

for n = [10 50 100 200 500 1000 10000 100000]
    S = InformationSource(n) ;
    bs = SourceEncoder(n, S) ;
    EX = length(bs) / n ;
    disp(EX) ;
end

%%
function S = InformationSource(n)
S = "" ;
    for i = [1:1:n]
        seed = floor(rand() * 32) ;
        if(seed >=0 && seed < 16)
            S = [S,'a'] ;
        elseif(seed >=16 && seed < 24)
            S = [S,'b'] ;
        elseif(seed >= 24 && seed < 28)
            S = [S , 'c'] ;
        elseif(seed >= 28 && seed < 30)
            S = [S , 'd'] ;
        elseif(seed == 30)
            S = [S , 'e'] ;
        elseif(seed == 31)
            S = [S , 'f'] ;
        end
    end
end
function [Encode] = SourceEncoder(n, S)
Encode = [] ;
    for i = [1:1:length(S)]
        if(S(i) == 'a')
            Encode = [Encode,1] ;
        elseif(S(i) == 'b')
             Encode = [Encode,0,1] ;
        elseif(S(i) == 'c')
             Encode = [Encode,0,0,1] ;
        elseif(S(i) == 'd')
             Encode = [Encode,0,0,0,1] ;
        elseif(S(i) == 'e')
             Encode = [Encode,0,0,0,0,1] ;
        elseif(S(i) == 'f')
             Encode = [Encode,0,0,0,0,0] ;
        end
    end
end
    

